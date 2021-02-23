from .base import Dataset
from .coco import CocoDetection
from .synthetic import SynDetection2D

__all__ = [
    "CocoDetection",
    "Dataset",
    "SynDetection2D",
]
