import numpy as np
import torch
from pytest import approx

from datasetinsights.evaluation_metrics import IoU


def get_y_true_y_pred():
    # Generate an image with labels 0 (background), 1, 2
    # 3 classes:
    y_true = np.zeros((30, 30), dtype=np.int)
    y_true[1:11, 1:11] = 1
    y_true[15:25, 15:25] = 2

    y_pred = np.zeros((30, 30), dtype=np.int)
    y_pred[5:15, 1:11] = 1
    y_pred[20:30, 20:30] = 2

    return y_true, y_pred


def compute_th_y_true_y_logits(y_true, y_pred):
    # Create torch.tensor from numpy
    th_y_true = torch.from_numpy(y_true).unsqueeze(0)
    # Create logits torch.tensor:
    num_classes = max(np.max(y_true), np.max(y_pred)) + 1
    y_probas = np.ones((num_classes,) + y_true.shape) * -10
    for i in range(num_classes):
        y_probas[i, (y_pred == i)] = 720
    th_y_logits = torch.from_numpy(y_probas).unsqueeze(0)

    return th_y_true, th_y_logits


def test_iou():
    y_true, y_pred = get_y_true_y_pred()
    th_y_true, th_y_logits = compute_th_y_true_y_logits(y_true, y_pred)

    true_res = [0, 0, 0]
    for index in range(3):
        bin_y_true = y_true == index
        bin_y_pred = y_pred == index
        intersection = bin_y_true & bin_y_pred
        union = bin_y_true | bin_y_pred
        true_res[index] = intersection.sum() / union.sum()

    iou_metric = IoU(num_classes=3)

    # Update metric
    output = (th_y_logits, th_y_true)
    iou_metric.update(output)

    res = iou_metric.compute().numpy()

    assert res == approx(true_res)
