import cv2
import matplotlib
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import time
import json

class DepthAnythingNode(Node):
    def __init__(self):
        super().__init__('depth_anything_node')

        self.declare_parameter('backend', 'onnx') # 'onnx' / 'torch'
        self.declare_parameter('model_path', '')
        self.declare_parameter('device', 'cuda') # 'cpu' / 'cuda'
        self.declare_parameter('invert', False)
        self.declare_parameter('processing_period', 0.0) # seconds
        self.declare_parameter('input_topic', '/camera/image_raw')
        self.declare_parameter('output_topic', '/depth')
        self.declare_parameter('show_result', False)
        self.declare_parameter('colored_depth', False)
        self.declare_parameter('use_scale', False)
        self.declare_parameter('scale', 4.25)
        self.declare_parameter('use_fp16', True)
        self.declare_parameter('use_compile', True)
        self.declare_parameter('input_size_w', 0)  # 0 = use original width
        self.declare_parameter('input_size_h', 0)  # 0 = use original height
        self.declare_parameter('max_depth', 0.0)  # 0 = no filtering, >0 = filter out points farther than this (meters)
        self.declare_parameter('performance_metrics_enabled', False)
        self.declare_parameter('performance_log_period', 2.0)
        self.declare_parameter('performance_metrics_topic', '/depth_anything/perf')

        backend_type = self.get_parameter('backend').value
        model_path = self.get_parameter('model_path').value
        device = self.get_parameter('device').value
        self.invert = self.get_parameter('invert').value
        self.processing_period = self.get_parameter('processing_period').value
        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.show_result = self.get_parameter('show_result').value
        
        # Initialize timing variables
        self.last_processing_time = 0.0
        if self.processing_period > 0.0:
            self.get_logger().info(f'Processing period set to {self.processing_period:.2f} seconds')
        else:
            self.get_logger().info('Processing every frame (no throttling)')
        self.colored_depth = self.get_parameter('colored_depth').value
        self.use_scale = self.get_parameter('use_scale').value
        self.scale = self.get_parameter('scale').value
        use_fp16 = self.get_parameter('use_fp16').value
        use_compile = self.get_parameter('use_compile').value
        input_size_w = self.get_parameter('input_size_w').value
        input_size_h = self.get_parameter('input_size_h').value
        self.max_depth = self.get_parameter('max_depth').value
        self.performance_metrics_enabled = self.get_parameter('performance_metrics_enabled').value
        self.performance_log_period = self.get_parameter('performance_log_period').value
        self.performance_metrics_topic = self.get_parameter('performance_metrics_topic').value

        if not model_path:
            self.get_logger().error('Model path is empty! Please set model_path parameter.')
            raise ValueError('Model path cannot be empty')
        
        try:
            if backend_type == 'onnx':
                from .backends.onnx_runner import ONNXRunner
                self.runner = ONNXRunner(
                    model_path, device, self.get_logger(),
                    input_size_w=input_size_w, input_size_h=input_size_h
                )
            elif backend_type == 'torch':
                from .backends.torch_runner import TorchRunner
                self.runner = TorchRunner(
                    model_path, device, self.get_logger(),
                    use_fp16=use_fp16, use_compile=use_compile,
                    input_size_w=input_size_w, input_size_h=input_size_h
                )
            else:
                self.get_logger().error(f'Invalid backend: {backend_type}. Valid options: onnx, torch')
                raise ValueError(f"Invalid backend: {backend_type}")
        except Exception as e:
            self.get_logger().error(f'Failed to load backend: {str(e)}')
            raise

        self.bridge = CvBridge()
        
        self.sub = self.create_subscription(
            Image,
            input_topic,
            self.image_callback,
            1
        )
        self.get_logger().info(f'Subscribed to image topic: {input_topic}')
        
        self.pub = self.create_publisher(Image, output_topic, 10)
        self.get_logger().info(f'Publishing to depth topic: {output_topic}')
        self.perf_pub = None
        if self.performance_metrics_enabled:
            self.perf_pub = self.create_publisher(String, self.performance_metrics_topic, 10)
            self._reset_perf_window()
            self.get_logger().info(
                f'Performance metrics enabled: topic={self.performance_metrics_topic}, period={self.performance_log_period:.2f}s'
            )
        self.get_logger().info('Initialization complete. Waiting for images...')

    def image_callback(self, msg):
        try:
            cb_started_at = None
            if self.performance_metrics_enabled:
                cb_started_at = time.perf_counter()
                self._perf_received_frames += 1

            # Check if enough time has passed since last processing
            current_time = time.time()
            if self.processing_period > 0.0:
                time_elapsed = current_time - self.last_processing_time
                if time_elapsed < self.processing_period:
                    self._finalize_perf_sample(cb_started_at, skipped=True)
                    return  # Skip this frame
            
            self.last_processing_time = current_time
            
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            infer_started_at = time.perf_counter() if self.performance_metrics_enabled else None
            depth_map = self.runner.infer(cv_image)
            infer_elapsed = 0.0
            if infer_started_at is not None:
                infer_elapsed = time.perf_counter() - infer_started_at
            if self.use_scale:
                metric_depth = self.scale / np.clip(depth_map, 1e-4, None)
            else:
                metric_depth = depth_map
            
            # Filter out points farther than max_depth
            if self.max_depth > 0.0:
                metric_depth = np.where(metric_depth > self.max_depth, 0.0, metric_depth)
            
            depth_msg = self.bridge.cv2_to_imgmsg(metric_depth.astype(np.float32), encoding='32FC1')
            depth_msg.header = msg.header
            self.pub.publish(depth_msg)
            self._finalize_perf_sample(cb_started_at, processed=True, infer_dt=infer_elapsed)

            if self.show_result:
                cv2.imshow('DepthAnything Input', cv_image)
                if self.invert:
                    depth_map = np.max(depth_map) - depth_map
                depth_display = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255.0
                depth_display = depth_display.astype(np.uint8)
                if self.colored_depth:
                    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
                    depth_display = (cmap(depth_display)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
                cv2.imshow('DepthAnything Depth', depth_display)
                cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}', throttle_duration_sec=1.0)

    def _reset_perf_window(self):
        now = time.perf_counter()
        self._perf_window_started_at = now
        self._perf_received_frames = 0
        self._perf_processed_frames = 0
        self._perf_skipped_frames = 0
        self._perf_infer_time = 0.0
        self._perf_callback_time = 0.0
        self._perf_prev_cpu_time = time.process_time()
        self._perf_prev_cpu_wall = now

    def _finalize_perf_sample(self, callback_started_at, processed=False, infer_dt=0.0, skipped=False):
        if not self.performance_metrics_enabled or callback_started_at is None:
            return

        if processed:
            self._perf_processed_frames += 1
            self._perf_infer_time += infer_dt
        if skipped:
            self._perf_skipped_frames += 1

        self._perf_callback_time += max(0.0, time.perf_counter() - callback_started_at)
        self._publish_perf_if_due()

    def _publish_perf_if_due(self):
        if not self.performance_metrics_enabled:
            return

        now = time.perf_counter()
        window_dt = now - self._perf_window_started_at
        if window_dt < max(0.2, self.performance_log_period):
            return

        cpu_now = time.process_time()
        cpu_wall_dt = max(1e-6, now - self._perf_prev_cpu_wall)
        cpu_time_dt = max(0.0, cpu_now - self._perf_prev_cpu_time)
        cpu_percent = 100.0 * cpu_time_dt / cpu_wall_dt

        infer_ms = 0.0
        if self._perf_processed_frames > 0:
            infer_ms = 1000.0 * self._perf_infer_time / self._perf_processed_frames

        callback_ms = 0.0
        if self._perf_received_frames > 0:
            callback_ms = 1000.0 * self._perf_callback_time / self._perf_received_frames

        payload = {
            'node': 'depth_anything_node',
            'window_sec': round(window_dt, 3),
            'input_fps': round(self._perf_received_frames / window_dt, 3),
            'processed_fps': round(self._perf_processed_frames / window_dt, 3),
            'skip_fps': round(self._perf_skipped_frames / window_dt, 3),
            'infer_ms_avg': round(infer_ms, 3),
            'callback_ms_avg': round(callback_ms, 3),
            'process_cpu_percent': round(cpu_percent, 2),
            'processing_period': round(float(self.processing_period), 4),
        }

        if self.perf_pub is not None:
            msg = String()
            msg.data = json.dumps(payload, ensure_ascii=True)
            self.perf_pub.publish(msg)

        self.get_logger().info(f'PERF {msg.data if self.perf_pub is not None else json.dumps(payload, ensure_ascii=True)}')
        self._reset_perf_window()

def main(args=None):
    rclpy.init(args=args)
    node = DepthAnythingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()