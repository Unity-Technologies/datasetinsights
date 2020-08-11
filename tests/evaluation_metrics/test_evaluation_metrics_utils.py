from unittest.mock import patch

import numpy as np

from datasetinsights.evaluation_metrics.metrics_utils import (
    filter_pred_bboxes, mean_metrics_over_iou
)


@patch("datasetinsights.evaluation_metrics.AveragePrecision.compute")
def test_mean_metrics_over_iou(mock_ap_compute):
    iou_threshoulds = np.linspace(
        0.5, 0.95, np.round((0.95 - 0.5) / 0.05) + 1, endpoint=True
    )
    metric_per_iou = [AveragePrecision(iou) for iou in iou_threshoulds]
    mean_metrics_over_iou(metric_per_iou)
    



def test_filter_pred_bboxes():
