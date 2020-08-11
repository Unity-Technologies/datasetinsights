r"""Average Quaternion error metric.
"""

import numpy as np

from .base import EvaluationMetric
from .exceptions import NoSampleError


class AverageEulerError(EvaluationMetric):
    """Average Mean Square Error metric.

    The metric is defined for prediction loss.

    Attributes:
        sum_of_average_mean_square_error (float): the sum of AMSE
        for all the images in a branch
        num_samples (int): the number of samples in all mini-batches
    """

    def __init__(self):
        self.sum_of_average_quaternion_error = 0
        self.num_samples = 0

    def reset(self):
        self.sum_of_average_quaternion_error = 0
        self.num_samples = 0

    def complementary_rotation(self, y, y_pred):
        y_pred = np.mod(y_pred, 360)
        y_max, y_min = max(y, y_pred), min(y, y_pred)
        if (y_max - y_min) > 180: 
            y_max = 360 - y_max 
            return y_max + y_min 
        return y_max - y_min 

    def update(self, output):
        y_pred, y = output
        if y.shape != y_pred.shape:
            raise ValueError(f'The shapes of real data {y.shape} and predicted data {y_pred.shape} \
                               should be the same.')

        self.num_samples += y.shape[0]
        self.sum_of_average_quaternion_error += np.sum(self.complementary_rotation(y, y_pred))

    def compute(self):
        if self.num_samples == 0:
            raise NoSampleError(
                "AverageMeanSquareError must have at least one example \
                before it can be computed."
            )
        return self.sum_of_average_quaternion_error / self.num_samples
