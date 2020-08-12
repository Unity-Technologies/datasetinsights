"""Dummy dataset for testing."""

import numpy as np
from PIL import Image

from datasetinsights.datasets import Dataset
from datasetinsights.io.bbox import BBox2D


class DummyDetection2D(Dataset):
    """Dummy dataset creation with 2 images for testing.

    Attributes:
        images: list of images of dataset
        bboxes: target bounding boxes
        label_mappings: class label mappings
        transform: transform the images and bounding boxes

    """

    def __init__(self, image_size=(256, 256), transform=None):
        """initiate dataset class.

        Args:
            image_size : size of images you want generate
            transform : transform the images and bounding boxes
        """
        self.images = [
            Image.fromarray(np.random.random(image_size), "L"),
            Image.fromarray(np.random.random(image_size), "L"),
        ]
        self.bboxes = [
            [
                BBox2D(label=1, x=10, y=20, w=30, h=40),
                BBox2D(label=2, x=50, y=50, w=10, h=10),
            ],
            [
                BBox2D(label=1, x=30, y=40, w=20, h=20),
                BBox2D(label=2, x=20, y=10, w=40, h=10),
            ],
        ]
        self.label_mappings = {"1": "car", "2": "bike"}

        self.transform = transform

    def __len__(self):
        """get dataset length."""
        return len(self.images)

    def __getitem__(self, index):
        """get data point."""
        image, bboxes = self.images[index], self.bboxes[index]

        if self.transform:
            image, bboxes = self.transform(image, bboxes)
        return image, bboxes

    def download(self):
        """pass download."""
        pass
