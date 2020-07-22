r"""Average Log10 Error metric.

The average relative error can be described as:

.. math::
    (\delta_i):\%\:of\:y_p\:s.t.\:
    max(\frac{y_p}{\hat{y_p}})=\delta<thr\:for\:thr=1.25,1.25^2,1.25^3
"""
import numpy as np

from .base import EvaluationMetric
from .exceptions import NoSampleError


class ThresholdAccuracy(EvaluationMetric):
    """Threshold accuracy metric.

    The metric is defined for grayscale depth images.

    Attributes:
        sum_of_threshold_acc (int): the sum of threshold accuracies for all the
         images in a branch
        num_samples (int): the number of samples in all mini-batches
    """

    def __init__(self, threshold):
        self.threshold = threshold
        self.sum_of_threshold_acc = 0
        self.num_samples = 0

    def reset(self):
        self.sum_of_threshold_acc = 0
        self.num_samples = 0

    def update(self, output):
        y_pred, y = output
        if y.shape != y_pred.shape:
            raise ValueError(
                f"The shapes of real data {y.shape} and predicted data "
                f"{y_pred.shape} should be the same."
            )

        thresh_scores = np.maximum(y / (y_pred + 1e-15), y_pred / (y + 1e-15))
        self.num_samples += y.shape[0]
        self.sum_of_threshold_acc += (
            np.sum(thresh_scores < self.threshold) / y.size * y.shape[0]
        )

    def compute(self):
        if self.num_samples == 0:
            raise NoSampleError(
                "ThresholdAccuracy must have at least one example before it "
                "can be computed."
            )
        return self.sum_of_threshold_acc / self.num_samples
