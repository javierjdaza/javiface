import onnxruntime as ort
import numpy as np
from PIL import Image
# from torchvision import transforms


class FaceVerifier:

    PROVIDERS = [
        ('CUDAExecutionProvider',   'CUDA'),
        ('CoreMLExecutionProvider', 'CoreML'),
        ('CPUExecutionProvider',    'CPU'),
    ]

    # TRANSFORM = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    # ])
    
    METADATA = {
        "threshold_youden": 0.21655870974063873,
        "threshold_at_far_1e3": 0.2136,
        "threshold_at_far_1e4": 0.2629,
        "threshold_at_far_1e5": 0.3095,
        "threshold_at_far_1e6": 0.3242,
        "TAR@FAR=1e-3": 0.993,
        "TAR@FAR=1e-4": 0.9909,
        "TAR@FAR=1e-5": 0.9895,
        "TAR@FAR=1e-6": 0.9891,
        "embedding_dim": 512,
        "input_size": [224, 224],
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
        "similarity": "cosine"
    }


    def __init__(self, onnx_path):
        available = ort.get_available_providers()
        providers = [p for p, _ in self.PROVIDERS if p in available]
        device = next(name for p, name in self.PROVIDERS if p in available)
        self.threshold = self.METADATA["threshold_at_far_1e4"] 
        
        self.sess = ort.InferenceSession(onnx_path, providers = providers)
        self.device = device
        print(f'JaviFace loaded — provider: {device}')

    def transform(self, image_pillow):
        image = image_pillow.resize((224, 224))
        image = np.array(image, dtype=np.float32) / 255.0        # [0, 255] → [0.0, 1.0]
        image = (image - self.METADATA["normalize_mean"]) / self.METADATA["normalize_std"]       
        image = image.transpose(2, 0, 1)                          # HWC -> CHW  (requerido por ONNX/PyTorch)
        image = image.astype(np.float32)
        return image
    
    
    def get_embedding(self, image_pillow):
        tensor = self.transform(image_pillow.convert('RGB'))      # (3, 224, 224)
        tensor = np.expand_dims(tensor, axis=0)              # (1, 3, 224, 224)
        return self.sess.run(['embedding'], {'image': tensor})[0][0]

    def compare(self, image_pillow_1, image_pillow_2, threshold = None):
        if threshold is None:
            threshold = self.threshold

        embedding_1 = self.get_embedding(image_pillow_1)
        embedding_2 = self.get_embedding(image_pillow_2)
        
        # Cosine Similarity
        similarity = float(np.dot(embedding_1, embedding_2))
        
        return {
            'similarity':  similarity,
            'same_person': similarity >= threshold,
        }
      
    def get_embedding_batch(self, images_pillow: list):
        tensors = np.stack([
            self.transform(img.convert('RGB'))
            for img in images_pillow
        ])  # (N, 3, 224, 224)
        return self.sess.run(['embedding'], {'image': tensors})[0]  # (N, embedding_dim)

    def compare_batch(self, images_a: list, images_b: list, threshold = None):
        if threshold is None:
            threshold = self.threshold
        
        embeddings_a = self.get_embedding_batch(images_a)  # (N, D)
        embeddings_b = self.get_embedding_batch(images_b)  # (N, D)
        
        # Cosine similarity fila a fila
        similarities = (embeddings_a * embeddings_b).sum(axis=1)  # (N,)

        results = []
        for sim in similarities:
            sim = float(sim)
            results.append({'similarity' : sim, 'same_person': sim >= threshold})

        return results