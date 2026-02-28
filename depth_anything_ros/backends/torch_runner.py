import cv2
import numpy as np
from .base_runner import BaseRunner

import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

class TorchRunner(BaseRunner):
    def __init__(self, model_path: str, device: str = 'cuda', logger=None, 
                 use_fp16: bool = True, use_compile: bool = True, input_size: int = 0):
        super().__init__(model_path, input_size)
        self.logger = logger
        self.use_fp16 = use_fp16 and device.lower() == 'cuda'
        self.use_compile = use_compile
        
        if self.logger:
            self.logger.info(f'Initializing PyTorch backend with model: {model_path}')
            self.logger.info(f'Requested device: {device}')
            self.logger.info(f'FP16 enabled: {self.use_fp16}')
            self.logger.info(f'torch.compile enabled: {self.use_compile}')
            self.logger.info(f'Input size: {input_size if input_size > 0 else "original"}')
        
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
                    self.use_fp16 = False  # Disable FP16 on CPU
            elif device_lower == 'cpu':
                self.device = torch.device("cpu")
                self.use_fp16 = False  # FP16 not efficient on CPU
            else:
                if self.logger:
                    self.logger.warn(f'Unknown device: {device}, defaulting to CPU')
                self.device = torch.device("cpu")
                self.use_fp16 = False
            
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
            
            # Convert to FP16 if enabled
            if self.use_fp16:
                self.model = self.model.half()
                if self.logger:
                    self.logger.info('Model converted to FP16 (half precision)')
            
            self.model.eval()
            
            # Apply torch.compile with reduce-overhead
            if self.use_compile and self.device.type == 'cuda':
                if self.logger:
                    self.logger.info('Compiling model with torch.compile (reduce-overhead)...')
                    self.logger.info('First inference will be slower due to compilation')
                try:
                    self.model = torch.compile(
                        self.model,
                        mode='reduce-overhead',
                        fullgraph=False,  # Allow graph breaks for compatibility
                    )
                    if self.logger:
                        self.logger.info('Model compiled successfully')
                except Exception as e:
                    if self.logger:
                        self.logger.warn(f'torch.compile failed: {e}, using eager mode')
                    self.use_compile = False
            
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
            # Resize input if input_size is set
            resized_image, orig_size = self.preprocess_resize(cv_image)
            orig_h, orig_w = orig_size
            
            rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            inputs = self.processor(images=rgb_image, return_tensors="pt").to(self.device)
            
            # Convert inputs to FP16 if enabled
            if self.use_fp16:
                inputs = {k: v.half() if v.dtype == torch.float32 else v 
                         for k, v in inputs.items()}

            with torch.no_grad():
                if self.device.type == 'cuda':
                    with torch.amp.autocast("cuda", enabled=self.use_fp16):
                        outputs = self.model(**inputs)
                        predicted_depth = outputs.predicted_depth
                else:
                    outputs = self.model(**inputs)
                    predicted_depth = outputs.predicted_depth

            depth = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1).float(),  # Ensure float32 for interpolation
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