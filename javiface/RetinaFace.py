"""
===================================================================================================================
This is my own implementation for RetinaFace Model using Resnet34 onnx file.
    Based on: https://github.com/yakhyo/retinaface-pytorch/tree/main
    Pre-Trained Model: https://github.com/yakhyo/retinaface-pytorch/releases/download/v0.0.1/retinaface_r34.onnx

@Author: javier.daza@mercadolibre.com.co
===================================================================================================================
"""

import numpy as np
import onnxruntime as ort
from PIL import Image
import cv2
import math
from itertools import product
import torch
from typing import Tuple

class RetinaFaceONNXInference:
    def __init__(
        self,
        model_path,
        conf_threshold=0.05,
        pre_nms_topk=5000,
        nms_threshold=0.4,
        post_nms_topk=750,
        vis_threshold=0.9
    ) -> None:
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.pre_nms_topk = pre_nms_topk
        self.nms_threshold = nms_threshold
        self.post_nms_topk = post_nms_topk
        self.vis_threshold = vis_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load ONNX model
        self.ort_session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.provider = self.ort_session.get_providers()
        print(f'RetinaFace loaded — provider: {"CUDA" if self.provider == "CUDAExecutionProvider" else "CPU"}')
        # Config for prior boxes
        self.cfg = {
            'name': 'resnet34',
            'min_sizes': [[16, 32], [64, 128], [256, 512]],
            'steps': [8, 16, 32],
            'variance': [0.1, 0.2],
            'clip': False,
            'loc_weight': 2.0,
            'batch_size': 32,
            'epochs': 100,
            'milestones': [70, 90],
            'image_size': 640,
            'pretrain': True,
            'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
            'in_channel': 64,
            'out_channel': 128
        }
        

    
    
    @staticmethod
    def preprocess_image(image, rgb_mean=(104, 117, 123)):
        image = np.float32(image)
        image -= rgb_mean
        image = image.transpose(2, 0, 1)  # HWC to CHW
        image = np.expand_dims(image, axis=0)  # Add batch dimension (1, C, H, W)
        return image

    
    @staticmethod
    def nms(dets, threshold):
        """
        Apply Non-Maximum Suppression (NMS) to reduce overlapping bounding boxes based on a threshold.
    
        Args:
            dets (numpy.ndarray): Array of detections with each row as [x1, y1, x2, y2, score].
            threshold (float): IoU threshold for suppression.
    
        Returns:
            list: Indices of bounding boxes retained after suppression.
        """
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]
    
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
    
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
    
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
    
            inds = np.where(ovr <= threshold)[0]
            order = order[inds + 1]
    
        return keep
    @staticmethod
    def decode(loc, priors, variances):
        """
        Decode locations from predictions using priors to undo
        the encoding done for offset regression at train time.
    
        Args:
            loc (tensor): Location predictions for loc layers, shape: [num_priors, 4]
            priors (tensor): Prior boxes in center-offset form, shape: [num_priors, 4]
            variances (list[float]): Variances of prior boxes
    
        Returns:
            tensor: Decoded bounding box predictions
        """
        # Compute centers of predicted boxes
        cxcy = priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:]
    
        # Compute widths and heights of predicted boxes
        wh = priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])
    
        # Convert center, size to corner coordinates
        boxes = torch.empty_like(loc)
        boxes[:, :2] = cxcy - wh / 2  # xmin, ymin
        boxes[:, 2:] = cxcy + wh / 2  # xmax, ymax
    
        return boxes
    
    @staticmethod
    def decode_landmarks(predictions, priors, variances):
        """
        Decode landmarks from predictions using prior boxes to reverse the encoding done during training.
    
        Args:
            predictions (tensor): Landmark predictions for localization layers.
                Shape: [num_priors, 10] where each prior contains 5 landmark (x, y) pairs.
            priors (tensor): Prior boxes in center-offset form.
                Shape: [num_priors, 4], where each prior has (cx, cy, width, height).
            variances (list[float]): Variances of the prior boxes to scale the decoded values.
    
        Returns:
            landmarks (tensor): Decoded landmark predictions.
                Shape: [num_priors, 10] where each row contains the decoded (x, y) pairs for 5 landmarks.
        """
    
        # Reshape predictions to [num_priors, 5, 2] to handle each pair (x, y) in a batch
        predictions = predictions.view(predictions.size(0), 5, 2)
    
        # Perform the same operation on all landmark pairs at once
        landmarks = priors[:, :2].unsqueeze(1) + predictions * variances[0] * priors[:, 2:].unsqueeze(1)
    
        # Flatten back to [num_priors, 10]
        landmarks = landmarks.view(landmarks.size(0), -1)
    
        return landmarks

    def infer(self, image_array):
        # Load and preprocess image
        # original_image = cv2.imread(image_path)
        original_image = image_array
        img_height, img_width, _ = original_image.shape
        image = self.preprocess_image(original_image)

        # Run ONNX model inference
        outputs = self.ort_session.run(None, {'input': image})
        loc, conf, landmarks = outputs[0].squeeze(0), outputs[1].squeeze(0), outputs[2].squeeze(0)

        # Generate anchor boxes
        priorbox = PriorBox(self.cfg, image_size=(img_height, img_width))
        priors = priorbox.generate_anchors()

        # Decode boxes and landmarks
        boxes = self.decode(torch.tensor(loc), priors, self.cfg['variance']).to(self.device)
        landmarks = self.decode_landmarks(torch.tensor(landmarks), priors, self.cfg['variance']).to(self.device)

        # Adjust scales for boxes and landmarks
        bbox_scale = torch.tensor([img_width, img_height] * 2, device=self.device)
        boxes = (boxes * bbox_scale).cpu().numpy()

        landmark_scale = torch.tensor([img_width, img_height] * 5, device=self.device)
        landmarks = (landmarks * landmark_scale).cpu().numpy()

        scores = conf[:, 1]  # Confidence scores for class 1 (face)

        # Filter by confidence threshold
        inds = scores > self.conf_threshold
        boxes, landmarks, scores = boxes[inds], landmarks[inds], scores[inds]

        # Sort by scores
        order = scores.argsort()[::-1][:self.pre_nms_topk]
        boxes, landmarks, scores = boxes[order], landmarks[order], scores[order]

        # Apply NMS
        detections = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = self.nms(detections, self.nms_threshold)
        detections, landmarks = detections[keep], landmarks[keep]

        # Keep top-k detections
        detections, landmarks = detections[:self.post_nms_topk], landmarks[:self.post_nms_topk]

        # Concatenate detections and landmarks
        return np.concatenate((detections, landmarks), axis=1), original_image


    # -- Get Face Fuction ---
    def crop_face_rf(self, image_pillow, vis_threshold = 0.8, expand_face_area=0.2):
        image_array = np.array(image_pillow)
        detections, landmarks = self.infer(image_array)
        
        if vis_threshold:
            detections = [d for d in detections if d[4] >= vis_threshold]
        else:
            detections = [d for d in detections if d[4] >= self.vis_threshold]
        
        if len(detections) > 0:
            best = max(detections, key=lambda d: (d[2] - d[0]) * (d[3] - d[1]))
            left, top, right, bottom = int(best[0]), int(best[1]), int(best[2]), int(best[3])
        
            w, h = image_pillow.size
            pw = (right - left) * expand_face_area
            ph = (bottom - top) * expand_face_area
            left   = max(0, int(left - pw))
            top    = max(0, int(top - ph))
            right  = min(w, int(right + pw))
            bottom = min(h, int(bottom + ph))
            
            return image_pillow.crop((left, top, right, bottom)), float(best[4])
        else:
            return None

class PriorBox:
    def __init__(self, cfg: dict, image_size: Tuple[int, int]) -> None:
        super().__init__()
        self.image_size = image_size
        self.clip = cfg['clip']
        self.steps = cfg['steps']
        self.min_sizes = cfg['min_sizes']
        self.feature_maps = [[
            math.ceil(self.image_size[0]/step), math.ceil(self.image_size[1]/step)] for step in self.steps
        ]
    
    def generate_anchors(self) -> torch.Tensor:
        """Generate anchor boxes based on configuration and image size"""
        anchors = []
        for k, (map_height, map_width) in enumerate(self.feature_maps):
            step = self.steps[k]
            for i, j in product(range(map_height), range(map_width)):
                for min_size in self.min_sizes[k]:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
    
                    dense_cx = [x * step / self.image_size[1] for x in [j+0.5]]
                    dense_cy = [y * step / self.image_size[0] for y in [i+0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]
    
        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


