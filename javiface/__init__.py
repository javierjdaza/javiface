import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from javiface.JaviFace import FaceVerifier as JaviFace
from javiface.RetinaFace import RetinaFaceONNXInference as RetinaFace

__all__ = ["JaviFace"]


"""
Javi Face.

Accurate Faces Comparison.
"""

__version__ = "0.0.8"
__author__ = 'Javier Javier Daza Olivella'