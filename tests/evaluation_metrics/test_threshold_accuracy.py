import numpy as np
from pytest import approx

from datasetinsights.evaluation_metrics import ThresholdAccuracy


def get_y_true_y_pred():
    # Generate two ground-truth images and two pred images with different depths
    y_true = np.ones((2, 2, 2, 3), dtype=np.double) * 10

    y_pred = np.ones((2, 2, 2, 3), dtype=np.double) * 10
    y_pred[0, 0, 0, 0] = 0
    y_pred[0, 1, 0, 1] = 19
    y_pred[1, 1, 1, 2] = 15

    return y_true, y_pred


def test_threshold_accraucy():
    y_true, y_pred = get_y_true_y_pred()

    threshold1, threshold2, threshold3 = 1.25, 1.25 ** 2, 1.25 ** 3
    a1_metric = ThresholdAccuracy(threshold1)
    a2_metric = ThresholdAccuracy(threshold2)
    a3_metric = ThresholdAccuracy(threshold3)

    # compute real metric
    n = y_true.size
    a1 = a2 = a3 = 0
    for i in range(y_true.shape[0]):
        for j in range(y_true.shape[1]):
            for k in range(y_true.shape[2]):
                for m in range(y_true.shape[3]):
                    thresh_score = max(
                        y_true[i][j][k][m] / (y_pred[i][j][k][m] + 1e-15),
                        y_pred[i][j][k][m] / (y_true[i][j][k][m] + 1e-15),
                    )
                    if thresh_score < threshold1:
                        a1 += 1
                    if thresh_score < threshold2:
                        a2 += 1
                    if thresh_score < threshold3:
                        a3 += 1

    true_res1 = a1 / n
    true_res2 = a2 / n
    true_res3 = a3 / n

    # Update metric
    output1 = (y_true[0], y_true[0])
    a1_metric.update(output1)
    a2_metric.update(output1)
    a3_metric.update(output1)

    output2 = (y_true[1], y_true[1])
    a1_metric.update(output2)
    a2_metric.update(output2)
    a3_metric.update(output2)
    res1 = a1_metric.compute()
    res2 = a2_metric.compute()
    res3 = a3_metric.compute()
    assert approx(1) == res1
    assert approx(1) == res2
    assert approx(1) == res3

    a1_metric.reset()
    a2_metric.reset()
    a3_metric.reset()

    output1 = (y_pred[0], y_true[0])
    a1_metric.update(output1)
    a2_metric.update(output1)
    a3_metric.update(output1)
    output2 = (y_pred[1], y_true[1])
    a1_metric.update(output2)
    a2_metric.update(output2)
    a3_metric.update(output2)
    res1 = a1_metric.compute()
    res2 = a2_metric.compute()
    res3 = a3_metric.compute()
    assert approx(true_res1) == res1
    assert approx(true_res2) == res2
    assert approx(true_res3) == res3
