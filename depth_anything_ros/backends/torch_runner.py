import cv2
import numpy as np
from .base_runner import BaseRunner

import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

class TorchRunner(BaseRunner):
    def __init__(self, model_path: str, device: str = 'cuda', logger=None):
        super().__init__(model_path)
        self.logger = logger
        
        if self.logger:
            self.logger.info(f'Initializing PyTorch backend with model: {model_path}')
            self.logger.info(f'Requested device: {device}')
        
        try:
            # Set device based on parameter
            device_lower = device.lower()
            if device_lower == 'cuda':
                if torch.cuda.is_available():
                    self.device = torch.device("cuda")
                else:
                    if self.logger:
                        self.logger.warn('CUDA requested but not available, falling back to CPU')
                    self.device = torch.device("cpu")
            elif device_lower == 'cpu':
                self.device = torch.device("cpu")
            else:
                if self.logger:
                    self.logger.warn(f'Unknown device: {device}, defaulting to CPU')
                self.device = torch.device("cpu")
            
            if self.logger:
                if self.device.type == 'cuda':
                    self.logger.info(f'Using GPU: {torch.cuda.get_device_name(0)}')
                    self.logger.info(f'CUDA version: {torch.version.cuda}')
                    self.logger.info(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
                else:
                    self.logger.info('Using CPU as requested')
            
            if self.logger:
                self.logger.info('Loading image processor...')
            self.processor = AutoImageProcessor.from_pretrained(self.model_path)
            
            if self.logger:
                self.logger.info('Loading depth estimation model...')
            self.model = AutoModelForDepthEstimation.from_pretrained(self.model_path).to(self.device)
            self.model.eval()
            
            if self.logger:
                total_params = sum(p.numel() for p in self.model.parameters())
                self.logger.info(f'Model loaded with {total_params:,} parameters')
                self.logger.info(f'PyTorch model initialization complete')
                
        except Exception as e:
            error_msg = f'Failed to load PyTorch model: {str(e)}'
            if self.logger:
                self.logger.error(error_msg)
            raise

    def infer(self, cv_image):
        try:
            orig_h, orig_w = cv_image.shape[:2]
            
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            inputs = self.processor(images=rgb_image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                predicted_depth = outputs.predicted_depth

            depth = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=(orig_h, orig_w),
                mode="bicubic",
                align_corners=False,
            )

            depth_map = depth.squeeze().cpu().numpy()
            
            return depth_map
            
        except Exception as e:
            if self.logger:
                self.logger.error(f'PyTorch inference failed: {str(e)}')
            raise