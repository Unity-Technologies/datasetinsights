r"""Average Log10 Error metric.

The average log10 error can be described as:

.. math::
    \frac{1}{n}\sum_{p}^{n} |log_{10}(y_p)-log_{10}(\hat{y_p})|

"""
import numpy as np

from .base import EvaluationMetric
from .exceptions import NoSampleError


class AverageLog10Error(EvaluationMetric):
    """Average Log10 Error metric.

    The metric is defined for grayscale depth images.

    Attributes:
        sum_of_log10_error (float): the sum of the log10 errors for all the
        images in a branch
        num_samples (int): the number of samples in all mini-batches
    """

    def __init__(self):
        self.sum_of_log10_error = 0
        self.num_samples = 0

    def reset(self):
        self.sum_of_log10_error = 0
        self.num_samples = 0

    def update(self, output):
        y_pred, y = output
        if y.shape != y_pred.shape:
            raise ValueError(
                f"The shapes of real data {y.shape} and predicted data "
                f"{y_pred.shape} should be the same."
            )

        self.num_samples += y.shape[0]
        self.sum_of_log10_error += (
            abs(np.log10(y + 1e-15) - np.log10(y_pred + 1e-15)).mean().item()
            * y.shape[0]
        )

    def compute(self):
        if self.num_samples == 0:
            raise NoSampleError(
                "AverageLog10Error must have at least one example before it "
                "can be computed."
            )
        return self.sum_of_log10_error / self.num_samples
