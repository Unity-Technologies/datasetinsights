from .base import Dataset
from .cityscapes import Cityscapes
from .coco import CocoDetection
from .groceries_real import GoogleGroceriesReal, GroceriesReal
from .nyudepth import NyuDepth
from .synthetic import SynDetection2D

__all__ = [
    "Cityscapes",
    "CocoDetection",
    "Dataset",
    "GroceriesReal",
    "GoogleGroceriesReal",
    "NyuDepth",
    "SynDetection2D",
]
