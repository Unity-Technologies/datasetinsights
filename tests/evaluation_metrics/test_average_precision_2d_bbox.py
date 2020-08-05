from unittest.mock import MagicMock, patch

from pytest import approx

from datasetinsights.evaluation_metrics import (
    AveragePrecisionBBox2D,
    MeanAveragePrecision,
)


def get_mini_batches(get_gt_pred_bbox):
    gt_bboxes, pred_bboxes = get_gt_pred_bbox

    mini_batch1 = [
        [gt_bboxes[0], pred_bboxes[0]],
        [gt_bboxes[1], pred_bboxes[1]],
    ]
    mini_batch2 = [
        [gt_bboxes[2], pred_bboxes[2]],
        [gt_bboxes[3], pred_bboxes[3]],
    ]
    mini_batch3 = [[gt_bboxes[4], pred_bboxes[4]]]
    return (mini_batch1, mini_batch2, mini_batch3)


def test_average_precision_2d_bbox(get_gt_pred_bbox):
    mini_batch1, mini_batch2, mini_batch3 = get_mini_batches(get_gt_pred_bbox)

    # test iou threshold = 0.5
    ap_metrics = AveragePrecisionBBox2D(iou_threshold=0.5)
    ap_metrics.update(mini_batch1)
    ap_metrics.update(mini_batch2)
    ap_metrics.update(mini_batch3)

    res = ap_metrics.compute()

    assert approx(res["car"], rel=1e-4) == 0.2257
    assert approx(res["pedestrian"], rel=1e-4) == 0.5
    assert approx(res["bike"], rel=1e-4) == 0

    # test reset function
    ap_metrics.reset()
    res = ap_metrics.compute()
    assert res == {}


@patch(
    "datasetinsights.evaluation_metrics.mean_average_precision"
    ".AveragePrecisionBBox2D.update"
)
@patch(
    "datasetinsights.evaluation_metrics.mean_average_precision"
    ".AveragePrecisionBBox2D.compute"
)
def test_mean_average_precision(mock_compute, mock_update, get_gt_pred_bbox):
    mini_batches = get_mini_batches(get_gt_pred_bbox)

    # test iou threshold = 0.5:0.05:0.95
    map_metrics = MeanAveragePrecision()

    for mini_batch in mini_batches:
        map_metrics.update(mini_batch)

    map_metrics.compute()
    assert mock_update.call_count == len(map_metrics.iou_thresholds) * len(
        mini_batches
    )
    assert mock_compute.call_count == len(map_metrics.iou_thresholds)
