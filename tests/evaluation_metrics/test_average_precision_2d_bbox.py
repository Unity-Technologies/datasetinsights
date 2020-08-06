from unittest.mock import patch

from pytest import approx

from datasetinsights.evaluation_metrics import (
    AveragePrecision50IOU,
    AveragePrecisionBBox2D,
    MeanAveragePrecision,
    MeanAveragePrecision50IOU,
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
    mini_batches = get_mini_batches(get_gt_pred_bbox)

    # test iou threshold = 0.5
    ap_metrics = AveragePrecisionBBox2D(iou_threshold=0.5)
    for mini_batch in mini_batches:
        ap_metrics.update(mini_batch)

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
    ".AveragePrecisionBBox2D.reset"
)
@patch(
    "datasetinsights.evaluation_metrics.mean_average_precision"
    ".AveragePrecisionBBox2D.update"
)
@patch(
    "datasetinsights.evaluation_metrics.mean_average_precision"
    ".AveragePrecisionBBox2D.compute"
)
def test_mean_average_precision(
    mock_compute, mock_update, mock_reset, get_gt_pred_bbox
):
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

    map_metrics.reset()
    mock_reset.assert_not_called()


@patch(
    "datasetinsights.evaluation_metrics.mean_average_precision"
    ".AveragePrecisionBBox2D.reset"
)
@patch(
    "datasetinsights.evaluation_metrics.average_precision_50iou"
    ".AveragePrecisionBBox2D.update"
)
@patch(
    "datasetinsights.evaluation_metrics.average_precision_50iou"
    ".AveragePrecisionBBox2D.compute"
)
def test_average_precision_50IOU(
    mock_compute, mock_update, mock_reset, get_gt_pred_bbox
):
    mini_batches = get_mini_batches(get_gt_pred_bbox)

    map_metrics = AveragePrecision50IOU()
    for mini_batch in mini_batches:
        map_metrics.update(mini_batch)

    map_metrics.compute()
    assert mock_update.call_count == len(mini_batches)
    mock_compute.assert_called()

    map_metrics.reset()
    mock_reset.assert_called()


@patch(
    "datasetinsights.evaluation_metrics.mean_average_precision"
    ".AveragePrecisionBBox2D.reset"
)
@patch(
    "datasetinsights.evaluation_metrics.mean_average_precision_50iou"
    ".AveragePrecisionBBox2D.update"
)
@patch(
    "datasetinsights.evaluation_metrics.mean_average_precision_50iou"
    ".AveragePrecisionBBox2D.compute"
)
def test_mean_average_precision_50IOU(
    mock_compute, mock_update, mock_reset, get_gt_pred_bbox
):
    mini_batches = get_mini_batches(get_gt_pred_bbox)

    map_metrics = MeanAveragePrecision50IOU()
    for mini_batch in mini_batches:
        map_metrics.update(mini_batch)

    map_metrics.compute()
    assert mock_update.call_count == len(mini_batches)
    mock_compute.assert_called()

    map_metrics.reset()
    mock_reset.assert_called()
