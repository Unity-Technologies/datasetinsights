import numpy as np
from PIL import Image

from datasetinsights.data.bbox import BBox2D
from datasetinsights.data.datasets import Dataset


class DummyDetection2D(Dataset):
    def __init__(
        self, image_size=(256, 256), transform=None, label_mappings=None
    ):
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
        self.label_mappings = label_mappings
        if not self.label_mappings:
            self.label_mappings = {"1": "car", "2": "bike"}

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image, bboxes = self.images[index], self.bboxes[index]

        if self.transform:
            image, bboxes = self.transform(image, bboxes)
        return image, bboxes

    def download(self):
        pass
