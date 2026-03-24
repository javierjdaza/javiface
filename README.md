# JaviFace 🎯

**Accurate face comparison powered by ONNX — selfie vs selfie, selfie vs ID, ID vs ID.**

[![PyPI version](https://img.shields.io/pypi/v/javiface)](https://pypi.org/project/javiface/)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## What is JaviFace?

JaviFace is a lightweight Python library for **face verification**. Given two face images, it tells you whether they belong to the same person — with scenario-specific thresholds calibrated for selfie and ID document comparisons.

Under the hood it runs two ONNX models:

| Component                  | Role                                  |
| -------------------------- | ------------------------------------- |
| **RetinaFace** (ResNet-34) | Face detection & crop                 |
| **FaceVerifier**           | 512-dim embedding + cosine similarity |

Inference runs on **CUDA**, **CoreML**, or **CPU** — automatically selected based on your hardware.

---

## Install

```bash
pip install javiface
```

**GPU acceleration (NVIDIA CUDA):** replace the default `onnxruntime` with `onnxruntime-gpu`:

```bash
pip install javiface
pip uninstall onnxruntime
pip install "onnxruntime-gpu>=1.22.0"
```

CPU / CoreML users don't need to do anything — `onnxruntime>=1.23.2` is installed automatically.

---

## Quick Start

```python
from PIL import Image
from javiface import JaviFace
from javiface import RetinaFace

# Load models
detector  = RetinaFace("retinaface_r34.onnx")
verifier  = JaviFace("javi_face_v1.onnx")

# Load images
img1 = Image.open("selfie.jpg")
img2 = Image.open("id_photo.jpg")

# Compare
# threshold = 0.2621 -> Selfie vs Selfie [default]
# threshold = 0.1838 -> Selfie vs ID document
# threshold = 0.1990 -> ID document vs ID document
result = verifier.compare(img1, img2, threshold = 0.2621)

print(result)
# {'similarity': 0.214, 'same_person': False}
```

---

## Model Metadata

| Parameter         | Value                 |
| ----------------- | --------------------- |
| Embedding dim     | 512                   |
| Input size        | 224 × 224             |
| Similarity metric | Cosine                |
| Normalize mean    | [0.485, 0.456, 0.406] |
| Normalize std     | [0.229, 0.224, 0.225] |

---

## Model Card — `javi_face_v1.onnx`

> ResNet-50 backbone + ArcFace head, trained from scratch on ~860 K face images across 94 K identities.

| Field             | Value                                    |
| ----------------- | ---------------------------------------- |
| **Architecture**  | ResNet-50 + ArcFace (m=0.5, s=64)        |
| **Embedding dim** | 512 — L2-normalized (unit hypersphere)   |
| **Training data** | 861 597 images · 94 261 identities       |
| **Export format** | ONNX (CUDA / CoreML / CPU)               |
| **Primary use**   | KYC — selfie vs ID document verification |

### Performance

| Scenario              | ROC-AUC | EER     | Precision | Recall  |
| --------------------- | ------- | ------- | --------- | ------- |
| Selfie vs Selfie      | 0.9993  | 0.485 % | 99.54 %   | 99.33 % |
| Selfie vs ID document | 0.9951  | 1.862 % | 97.31 %   | 97.47 % |
| ID vs ID              | 0.9930  | 2.228 % | 97.60 %   | 97.08 % |

Full training details and evaluation breakdown → [MODEL_CARD.md](MODEL_CARD.md)

---

## Thresholds

Choose the threshold that matches your use case. A similarity **≥ threshold** means same person.

| Scenario              | Threshold | Use when                  |
| --------------------- | --------- | ------------------------- |
| Selfie vs Selfie      | `0.2621`  | Comparing two live photos |
| Selfie vs ID document | `0.1838`  | KYC / onboarding flows    |
| ID vs ID              | `0.1990`  | Document deduplication    |

> **Lower threshold → stricter match.** ID photos have less variation than selfies, so the bar is lower.

---

## Hardware Acceleration

JaviFace automatically selects the best available provider:

```
FaceVerifier loaded — provider: CoreML   # macOS
FaceVerifier loaded — provider: CUDA     # NVIDIA GPU
FaceVerifier loaded — provider: CPU      # fallback
```

---

## Author

**Javier Daza** · [javierjdaza@gmail.com](mailto:javierjdaza@gmail.com) · [GitHub](https://github.com/javierjdaza/javiface/tree/main)

---

_MIT License_
