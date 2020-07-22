""" IoU evaluation metrics
"""
from ignite.metrics import ConfusionMatrix

from .base import EvaluationMetric


class IoU(EvaluationMetric):
    """ Intersection over Union (IoU) metric per class

    The metric is defined for a pair of grayscale semantic segmentation images.

    Args:
        num_classes: number of calsses in the ground truth image
        output_transform: function that transform output pair of images

    Attributes:
        cm (ignite.metrics.ConfusionMatrix): pytorch ignite confusion matrix
        object.
    """

    def __init__(self, num_classes, output_transform=lambda x: x):
        self.cm = ConfusionMatrix(
            num_classes=num_classes,
            average=None,
            output_transform=output_transform,
        )

    def reset(self):
        self.cm.reset()

    def update(self, output):
        self.cm.update(output)

    def compute(self):
        cm = self.cm.compute()
        iou = cm.diag() / (cm.sum(dim=1) + cm.sum(dim=0) - cm.diag() + 1e-15)

        return iou
