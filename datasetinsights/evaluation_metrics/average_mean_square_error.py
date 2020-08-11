r"""Average Mean Square Error metrics.

The Average mean square error can be described as:

.. math:: // TODO fix
    \{\frac{1}{n}\sum_{p}^{n}{(y_p-\hat{y_p})}^2}
"""

import numpy as np

from .base import EvaluationMetric
from .exceptions import NoSampleError


class AverageMeanSquareError(EvaluationMetric):
    """Average Mean Square Error metric.

    The metric is defined for prediction loss.

    Attributes:
        sum_of_average_mean_square_error (float): the sum of AMSE
        for all the images in a branch
        num_samples (int): the number of samples in all mini-batches
    """

    def __init__(self):
        self.sum_of_average_mean_square_error = 0
        self.num_samples = 0

    def reset(self):
        self.sum_of_average_mean_square_error = 0
        self.num_samples = 0

    def update(self, output):
        y_pred, y = output
        if y.shape != y_pred.shape:
            raise ValueError(f'The shapes of real data {y.shape} and predicted data {y_pred.shape} \
                               should be the same.')

        self.num_samples += y.shape[0]
        self.sum_of_average_mean_square_error += np.sum(
            np.mean(np.square(y - y_pred), axis=-1))

    def compute(self):
        if self.num_samples == 0:
            raise NoSampleError(
                "AverageMeanSquareError must have at least one example \
                before it can be computed."
            )
        return self.sum_of_average_mean_square_error / self.num_samples
