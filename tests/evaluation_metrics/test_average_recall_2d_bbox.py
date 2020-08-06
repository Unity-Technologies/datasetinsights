from unittest.mock import patch

from pytest import approx

from datasetinsights.evaluation_metrics import (
    AverageRecallBBox2D,
    MeanAverageRecall,
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


def test_average_recall_2d_bbox(get_gt_pred_bbox):
    mini_batches = get_mini_batches(get_gt_pred_bbox)

    # iou_threshold=0.5, max_detections=100
    ar_metrics = AverageRecallBBox2D(iou_threshold=0.5, max_detections=100)
    for mini_batch in mini_batches:
        ar_metrics.update(mini_batch)

    res = ar_metrics.compute()
    assert approx(res["car"], rel=1e-4) == 0.4
    assert approx(res["pedestrian"], rel=1e-4) == 0.6667
    assert approx(res["bike"], rel=1e-4) == 0

    # test reset function
    ar_metrics.reset()
    res = ar_metrics.compute()
    assert res == {}

    # iou_threshold=0.5, max_detections=1
    ar_metrics.iou_end = 0.95
    ar_metrics.max_detections = 100
    for mini_batch in mini_batches:
        ar_metrics.update(mini_batch)

    res = ar_metrics.compute()

    assert approx(res["car"], rel=1e-4) == 0.4
    assert approx(res["pedestrian"], rel=1e-4) == 0.66667
    assert approx(res["bike"], rel=1e-4) == 0

    ar_metrics.reset()


@patch(
    "datasetinsights.evaluation_metrics.mean_average_recall"
    ".AverageRecallBBox2D.reset"
)
@patch(
    "datasetinsights.evaluation_metrics.mean_average_recall"
    ".AverageRecallBBox2D.update"
)
@patch(
    "datasetinsights.evaluation_metrics.mean_average_recall"
    ".AverageRecallBBox2D.compute"
)
def test_mean_average_recall_2d_bbox(
    mock_compute, mock_update, mock_reset, get_gt_pred_bbox
):
    mini_batches = get_mini_batches(get_gt_pred_bbox)

    # test iou threshold = 0.5:0.05:0.95
    mar_metrics = MeanAverageRecall()

    for mini_batch in mini_batches:
        mar_metrics.update(mini_batch)

    mar_metrics.compute()
    assert mock_update.call_count == len(mar_metrics.iou_thresholds) * len(
        mini_batches
    )
    assert mock_compute.call_count == len(mar_metrics.iou_thresholds)

    mar_metrics.reset()
    mock_reset.assert_not_called()
