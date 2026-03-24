import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms


class FaceVerifier:

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

  def __init__(self, onnx_path):
      available = ort.get_available_providers()
      providers = [p for p, _ in self.PROVIDERS if p in available]
      device = next(name for p, name in self.PROVIDERS if p in available)

      self.sess = ort.InferenceSession(onnx_path, providers = providers)
      self.device = device
      print(f'FaceVerifier loaded — provider: {device}')

  def get_embedding(self, image_pillow):
      img = image_pillow.convert('RGB')
      tensor = self.TRANSFORM(img).unsqueeze(0).numpy()
      return self.sess.run(['embedding'], {'image': tensor})[0][0]

  def compare(self, image_pillow_1, image_pillow_2, threshold):
      embedding_1 = self.get_embedding(image_pillow_1)
      embedding_2 = self.get_embedding(image_pillow_2)
      
      # Cosine Similarity
      similarity = float(np.dot(embedding_1, embedding_2))
      
      return {
          'similarity':  similarity,
          'same_person': similarity >= threshold,
      }