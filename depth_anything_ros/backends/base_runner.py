class BaseRunner:
    def __init__(self, model_path: str):
        self.model_path = model_path

    def infer(self, cv_image):
        raise NotImplementedError("Subclasses must implement this method")