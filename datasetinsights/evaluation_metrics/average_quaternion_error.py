r"""Average Quaternion error metric.
"""

import numpy as np

from .base import EvaluationMetric
from .exceptions import NoSampleError


class AverageQuaternionError(EvaluationMetric):
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

    def update(self, output):
        y_pred, y = output
        if y.shape != y_pred.shape:
            raise ValueError(f'The shapes of real data {y.shape} and predicted data {y_pred.shape} \
                               should be the same.')

        self.num_samples += y.shape[0]
        if 2 * np.square(np.dot(y, y_pred)) - 1 >= 0.999:
            self.sum_of_average_quaternion_error += 0
        elif 2 * np.square(np.dot(y, y_pred)) - 1 <= -0.999:
            self.sum_of_average_quaternion_error += 3.14
        else:
            self.sum_of_average_quaternion_error += np.sum(
                np.arccos((2 * np.square(np.dot(y, y_pred)) - 1)))

    def compute(self):
        if self.num_samples == 0:
            raise NoSampleError(
                "AverageMeanSquareError must have at least one example \
                before it can be computed."
            )
        return self.sum_of_average_quaternion_error / self.num_samples
