import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class DepthAnythingNode(Node):
    def __init__(self):
        super().__init__('depth_anything_node')

        self.declare_parameter('backend', 'onnx') # 'onnx' / 'torch'
        self.declare_parameter('model_path', '')
        self.declare_parameter('input_topic', '/camera/image_raw')
        self.declare_parameter('output_topic', '/depth/image_raw')
        self.declare_parameter('show_result', False)
        
        backend_type = self.get_parameter('backend').value
        model_path = self.get_parameter('model_path').value
        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.show_result = self.get_parameter('show_result').value
        
        if not model_path:
            self.get_logger().error('Model path is empty! Please set model_path parameter.')
            raise ValueError('Model path cannot be empty')
        
        try:
            if backend_type == 'onnx':
                from .backends.onnx_runner import ONNXRunner
                self.runner = ONNXRunner(model_path, self.get_logger())
            elif backend_type == 'torch':
                from .backends.torch_runner import TorchRunner
                self.runner = TorchRunner(model_path, self.get_logger())
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
            qos_profile_sensor_data
        )
        self.get_logger().info(f'Subscribed to image topic: {input_topic}')
        
        self.pub = self.create_publisher(Image, output_topic, 10)
        self.get_logger().info(f'Publishing to depth topic: {output_topic}')
        self.get_logger().info('Initialization complete. Waiting for images...')

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            depth_map = self.runner.infer(cv_image)
            depth_msg = self.bridge.cv2_to_imgmsg(depth_map, encoding='32FC1')
            depth_msg.header = msg.header
            self.pub.publish(depth_msg)

            if self.show_result:
                cv2.imshow('DepthAnything Input', cv_image)
                depth_display = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
                depth_display = depth_display.astype('uint8')
                cv2.imshow('DepthAnything Depth', depth_display)
                cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}', throttle_duration_sec=1.0)

def main(args=None):
    rclpy.init(args=args)
    node = DepthAnythingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()