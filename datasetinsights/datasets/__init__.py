from .base import Dataset
from .coco import CocoDetection
from .groceries_real import GroceriesReal
from .nyudepth import NyuDepth
from .synthetic import SynDetection2D

__all__ = [
    "CocoDetection",
    "Dataset",
    "GroceriesReal",
    "NyuDepth",
    "SynDetection2D",
]
