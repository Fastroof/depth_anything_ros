import cv2
import numpy as np
from .base_runner import BaseRunner
import os

import onnxruntime as ort

class ONNXRunner(BaseRunner):
    def __init__(self, model_path: str, device: str = 'cuda', logger=None):
        super().__init__(model_path)
        self.logger = logger
        self.device = device.lower()

        if self.logger:
            self.logger.info(f'Initializing ONNX Runtime with model: {model_path}')
            self.logger.info(f'Requested device: {self.device}')

        if not os.path.exists(model_path):
            error_msg = f'Model file not found: {model_path}'
            if self.logger:
                self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            # Set providers based on device parameter
            if self.device == 'cuda':
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            elif self.device == 'cpu':
                providers = ['CPUExecutionProvider']
            else:
                if self.logger:
                    self.logger.warn(f'Unknown device: {self.device}, defaulting to CPU')
                providers = ['CPUExecutionProvider']
            
            self.session = ort.InferenceSession(self.model_path, providers=providers)

            active_providers = self.session.get_providers()
            if self.logger:
                self.logger.info(f'Active ONNX Runtime providers: {active_providers}')
                if 'CUDAExecutionProvider' in active_providers:
                    self.logger.info('Using GPU acceleration (CUDA)')
                else:
                    if self.device == 'cuda':
                        self.logger.warn('CUDA requested but not available, using CPU')
                    else:
                        self.logger.info('Using CPU as requested')
            
            self.input_name = self.session.get_inputs()[0].name
            input_shape = self.session.get_inputs()[0].shape
            if self.logger:
                self.logger.info(f'Model input: {self.input_name}, shape: {input_shape}')
            
            self.input_size = (518, 518) 
            self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
            self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
            
            if self.logger:
                self.logger.info(f'ONNX model loaded successfully. Input size: {self.input_size}')
                
        except Exception as e:
            error_msg = f'Failed to load ONNX model: {str(e)}'
            if self.logger:
                self.logger.error(error_msg)
            raise

    def infer(self, cv_image):
        try:
            orig_h, orig_w = cv_image.shape[:2]

            img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.input_size)
            img = img.astype(np.float32) / 255.0
            img = (img - self.mean) / self.std
            img = img.transpose(2, 0, 1)
            img = np.expand_dims(img, axis=0)

            expected_rank = len(self.session.get_inputs()[0].shape)
            if expected_rank == 5:
                img = np.expand_dims(img, axis=0)

            depth_map = self.session.run(None, {self.input_name: img})[0]
            depth_map = depth_map.squeeze()
            depth_map = cv2.resize(depth_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            
            return depth_map
            
        except Exception as e:
            if self.logger:
                self.logger.error(f'ONNX inference failed: {str(e)}')
            raise