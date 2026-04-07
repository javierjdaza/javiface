# JaviFace 🎯

**Accurate selfie-to-selfie face verification.**

[![PyPI version](https://img.shields.io/pypi/v/javiface)](https://pypi.org/project/javiface/)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## What is JaviFace?

JaviFace is a lightweight Python library for **face verification**. Given two selfie images, it tells you whether they belong to the same person — with FAR-controlled thresholds calibrated on 1 M+ face images.

Under the hood it runs two models:

| Component                  | Format        | Role                                  |
| -------------------------- | ------------- | ------------------------------------- |
| **RetinaFace** (ResNet-50) | TensorFlow H5 | Face detection, alignment & crop      |
| **FaceVerifier**           | ONNX          | 512-dim embedding + cosine similarity |

**FaceVerifier** runs on **CUDA**, **CoreML**, or **CPU** — automatically selected based on your hardware.

---

## Install

```bash
pip install javiface
```

or

```bash
poetry add javiface
```

**Required:** `tensorflow` for RetinaFace detection. On TF ≥ 2.16 also install:

```bash
pip install tf-keras
```

**GPU acceleration (NVIDIA CUDA):** replace the default `onnxruntime` with `onnxruntime-gpu`:

```bash
pip uninstall onnxruntime
pip install "onnxruntime-gpu>=1.22.0"
```

---

## Quick Start

```python
from PIL import Image
from javiface import JaviFace, RetinaFace, RetinaFace34

# Load models
rf   = RetinaFace(model_path="retinaface.h5")
rf34 = RetinaFace34(model_path="retinaface_r34.onnx")  # fallback detector
jf   = JaviFace(onnx_path="javi_face_v1.onnx")

# Load images
img1 = Image.open("selfie1.jpg")
img2 = Image.open("selfie2.jpg")

# Detect & crop faces (PIL in → PIL out)
face1 = rf.get_faces(img_pillow=img1, threshold=0.2)
face2 = rf.get_faces(img_pillow=img2, threshold=0.2)

# Compare — choose threshold based on your FAR requirement:
# threshold = 0.2166 -> Youden index (balanced TPR/FPR) [default]
# threshold = 0.2136 -> FAR ≤ 10⁻³  | TAR = 99.30 %
# threshold = 0.2629 -> FAR ≤ 10⁻⁴  | TAR = 99.09 %
# threshold = 0.3095 -> FAR ≤ 10⁻⁵  | TAR = 98.95 %
# threshold = 0.3242 -> FAR ≤ 10⁻⁶  | TAR = 98.91 %
result = jf.compare(face1, face2, threshold=0.2629)  # TAR@FAR=1e-4

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

## Model Cards

### `retinaface.h5` — Face Detector

| Field            | Value                             |
| ---------------- | --------------------------------- |
| **Architecture** | ResNet-50 + FPN + SSH heads       |
| **Framework**    | TensorFlow / Keras                |
| **Output**       | Bounding boxes + 5 face landmarks |
| **Primary use**  | Face detection, alignment & crop  |

### `javi_face_v1.onnx` — Face Verifier

> ResNet-50 backbone + ArcFace head, trained from scratch on ~1 M selfie images across 232 K identities.

| Field             | Value                                        |
| ----------------- | -------------------------------------------- |
| **Architecture**  | ResNet-50 + ArcFace (m=0.5, s=64)            |
| **Embedding dim** | 512 — L2-normalized (unit hypersphere)        |
| **Training data** | 1 025 203 images · 232 659 identities        |
| **Parameters**    | 119 855 168 total · 118 410 240 trainable     |
| **Export format** | ONNX (CUDA / CoreML / CPU)                   |
| **Primary use**   | Selfie vs selfie face verification            |

### Performance — Selfie vs Selfie

Evaluated on 150 000 genuine pairs and 1 000 000 impostor pairs.

| Metric           | Value   |
| ---------------- | ------- |
| ROC-AUC          | 0.9987  |
| PR-AUC           | 0.9975  |
| EER              | 0.53 %  |
| Precision        | 99.41 % |
| Recall           | 99.29 % |
| TAR @ FAR = 10⁻³ | 99.30 % |
| TAR @ FAR = 10⁻⁴ | 99.09 % |
| TAR @ FAR = 10⁻⁵ | 98.95 % |
| TAR @ FAR = 10⁻⁶ | 98.91 % |

#### Similarity Distribution — Selfie vs Selfie

![Similarity Distribution](plots/similarity_distribution.png)

Full training details and evaluation breakdown → [MODEL_CARD.md](MODEL_CARD.md)

---

## Thresholds — Selfie vs Selfie

A similarity **≥ threshold** means same person. Choose based on your FAR requirement:

| Operating point        | Threshold | TAR     | Use when                                    |
| ---------------------- | --------- | ------- | ------------------------------------------- |
| Youden (balanced)      | `0.2166`  | —       | General use, balanced FAR/FRR               |
| FAR ≤ 10⁻³ (0.1 %)    | `0.2136`  | 99.30 % | Moderate security (consumer apps)           |
| FAR ≤ 10⁻⁴ (0.01 %)   | `0.2629`  | 99.09 % | Standard KYC / onboarding                  |
| FAR ≤ 10⁻⁵ (0.001 %)  | `0.3095`  | 98.95 % | High-security verification                  |
| FAR ≤ 10⁻⁶ (0.0001 %) | `0.3242`  | 98.91 % | Critical applications (banking, government) |

> **Higher threshold → stricter match → lower FAR.** Raising the bar reduces false accepts at the cost of a marginally lower true accept rate.

---

## Hardware Acceleration

**FaceVerifier** (ONNX) automatically selects the best available provider:

```
FaceVerifier loaded — provider: CoreML   # macOS
FaceVerifier loaded — provider: CUDA     # NVIDIA GPU
FaceVerifier loaded — provider: CPU      # fallback
```

**RetinaFace** (TensorFlow) uses whatever device TF has available. Set `TF_FORCE_GPU_ALLOW_GROWTH=true` (already set internally) to avoid reserving all VRAM.

---

## Author

**Javier Daza** · [javierjdaza@gmail.com](mailto:javierjdaza@gmail.com) · [GitHub](https://github.com/javierjdaza/javiface/tree/main)

---

_MIT License_
