from math import sqrt

import numpy as np
from pytest import approx

from datasetinsights.evaluation_metrics import RootMeanSquareError


def get_y_true_y_pred():
    # Generate two ground-truth images and two pred images with different
    # depths
    y_true = np.ones((2, 2, 2, 3), dtype=np.double) * 10

    y_pred = np.ones((2, 2, 2, 3), dtype=np.double) * 10
    y_pred[0, 0, 0, 0] = 0
    y_pred[0, 1, 0, 1] = 19
    y_pred[1, 1, 1, 2] = 15

    return y_true, y_pred


def test_root_mean_square_error():
    y_true, y_pred = get_y_true_y_pred()
    rmse_metric = RootMeanSquareError()
    # compute real metric
    num_pixels = y_true.size
    error = 0
    for i in range(y_true.shape[0]):
        for j in range(y_true.shape[1]):
            for k in range(y_true.shape[2]):
                for m in range(y_true.shape[3]):
                    error += np.square(y_true[i][j][k][m] - y_pred[i][j][k][m])

    true_res = sqrt((error / num_pixels).item())

    # Update metric
    output1 = (y_true[0], y_true[0])
    rmse_metric.update(output1)
    output2 = (y_true[1], y_true[1])
    rmse_metric.update(output2)
    res = rmse_metric.compute()
    assert approx(0) == res

    rmse_metric.reset()
    output1 = (y_pred[0], y_true[0])
    rmse_metric.update(output1)
    output2 = (y_pred[1], y_true[1])
    rmse_metric.update(output2)
    res = rmse_metric.compute()
    assert approx(true_res) == res
