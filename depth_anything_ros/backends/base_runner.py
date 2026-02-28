import cv2

class BaseRunner:
    def __init__(self, model_path: str, input_size: int = 0):
        self.model_path = model_path
        self.input_size = input_size  # 0 = use original, otherwise resize to NxN

    def preprocess_resize(self, cv_image):
        """Resize image if input_size is set. Returns (resized_image, original_size)"""
        orig_h, orig_w = cv_image.shape[:2]
        if self.input_size > 0:
            resized = cv2.resize(cv_image, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
            return resized, (orig_h, orig_w)
        return cv_image, (orig_h, orig_w)

    def postprocess_resize(self, depth_map, original_size):
        """Resize depth map back to original size"""
        orig_h, orig_w = original_size
        if depth_map.shape[0] != orig_h or depth_map.shape[1] != orig_w:
            return cv2.resize(depth_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        return depth_map

    def infer(self, cv_image):
        raise NotImplementedError("Subclasses must implement this method")