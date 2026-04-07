import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from javiface.JaviFace import FaceVerifier as JaviFace
from javiface.RetinaFace import RetinaFace
from javiface.RetinaFaceR34 import RetinaFaceONNXInference as RetinaFace34
from javiface.selfie_or_doc import SelfieOrDoc

__all__ = ["JaviFace", "RetinaFace", "RetinaFace34", "SelfieOrDoc"]


"""
Javi Face.

Accurate Faces Comparison.
"""

__version__ = "0.1.8"
__author__ = 'Javier Javier Daza Olivella'