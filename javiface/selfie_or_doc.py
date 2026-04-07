import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms


class SelfieOrDoc:

    PROVIDERS = [
      ('CUDAExecutionProvider',   'CUDA'),
      ('CoreMLExecutionProvider', 'CoreML'),
      ('CPUExecutionProvider',    'CPU'),
    ]

    TRANSFORM = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def __init__(self, model_path, threshold=0.9):
        available  = ort.get_available_providers()
        providers  = [p for p, _ in self.PROVIDERS if p in available]
        device = next(name for p, name in self.PROVIDERS if p in available)
        
        self.sess = ort.InferenceSession(model_path, providers=providers)
        self.threshold = threshold
        self.device = device
        print(f'SelfieOrDoc loaded — provider: {device}')
    
    def predict(self, image_pillow):
        tensor = self.TRANSFORM(image_pillow.convert('RGB')).unsqueeze(0).numpy()
        logit  = float(self.sess.run(['logit'], {'image': tensor})[0][0])
        prob   = 1 / (1 + np.exp(-logit))  # sigmoid
        return {
            'probability'     : float(round(prob,4)),
            'is_selfie': True if prob >= self.threshold else False,
            'label'    : 'selfie' if prob >= self.threshold else 'doc',
            }