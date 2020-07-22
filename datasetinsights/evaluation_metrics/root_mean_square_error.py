r"""Root Mean Square Error metrics.

The root mean square error can be described as:

.. math::
    \sqrt{\frac{1}{n}\sum_{p}^{n}{(y_p-\hat{y_p})}^2}
"""
from math import sqrt

import numpy as np

from .base import EvaluationMetric
from .exceptions import NoSampleError


class RootMeanSquareError(EvaluationMetric):
    """Root Mean Square Error metric.

    The metric is defined for grayscale depth images.

    Attributes:
        sum_of_root_mean_square_error (float): the sum of RMSE
        for all the images in a branch
        num_samples (int): the number of samples in all mini-batches
    """

    def __init__(self):
        self.sum_of_root_mean_square_error = 0
        self.num_samples = 0

    def reset(self):
        self.sum_of_root_mean_square_error = 0
        self.num_samples = 0

    def update(self, output):
        y_pred, y = output
        if y.shape != y_pred.shape:
            raise ValueError(
                f"The shapes of real data {y.shape} and predicted data {y_pred.shape} \
                               should be the same."
            )

        self.num_samples += y.shape[0]
        self.sum_of_root_mean_square_error += (
            np.square(y - y_pred).mean() * y.shape[0]
        )

    def compute(self):
        if self.num_samples == 0:
            raise NoSampleError(
                "RootMeanSquareError must have at least one example before it "
                "can be computed."
            )
        return sqrt(self.sum_of_root_mean_square_error / self.num_samples)
