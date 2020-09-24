r"""Average Relative Error metrics.

The average relative error can be described as:

.. math::
    \frac{1}{num\ samples}\sum_{p}^{num\ samples}\frac{|y_p-\hat{y_p}|}{y_p}
"""
import numpy as np

from .base import EvaluationMetric
from .exceptions import NoSampleError


class AverageRelativeError(EvaluationMetric):
    """Average Relative Error metric.

    The metric is defined for grayscale depth images.

    Attributes:
        sum_of_relative_error (float): the sum of the relative errors for all
        the images in a branch
        num_samples (int): the number of samples in all mini-batches
    """

    def __init__(self):
        self.sum_of_relative_error = 0
        self.num_samples = 0

    def reset(self):
        self.sum_of_relative_error = 0
        self.num_samples = 0

    def update(self, output):
        y_pred, y = output
        if y.shape != y_pred.shape:
            raise ValueError(
                f"The shapes of real data {y.shape} and predicted data "
                f"{y_pred.shape} should be the same."
            )

        self.num_samples += y.shape[0]
        self.sum_of_relative_error += (
            np.abs(y - y_pred) / (y + 1e-15)
        ).mean().item() * y.shape[0]

    def compute(self):
        if self.num_samples == 0:
            raise NoSampleError(
                "AverageRelativeError must have at least one example before"
                " it can be computed."
            )
        return self.sum_of_relative_error / self.num_samples
