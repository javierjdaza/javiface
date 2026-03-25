import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from javiface.JaviFace import FaceVerifier as JaviFace
from javiface import RetinaFace

__all__ = ["JaviFace", "RetinaFace"]


"""
Javi Face.

Accurate Faces Comparison.
"""

__version__ = "0.1.7"
__author__ = 'Javier Javier Daza Olivella'