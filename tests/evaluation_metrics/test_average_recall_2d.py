from unittest.mock import patch

from pytest import approx

from datasetinsights.evaluation_metrics import (
    AverageRecall,
    MeanAverageRecallAverageOverIOU,
)


def test_average_recall_2d_bbox(get_mini_batches):
    mini_batches = get_mini_batches

    # iou_threshold=0.5, max_detections=100
    ar_metrics = AverageRecall(iou_threshold=0.5, max_detections=100)
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


@patch("datasetinsights.evaluation_metrics.AverageRecall.reset")
@patch("datasetinsights.evaluation_metrics.AverageRecall.update")
@patch("datasetinsights.evaluation_metrics.AverageRecall.compute")
def test_mean_average_recall_2d_bbox(
    mock_compute, mock_update, mock_reset, get_mini_batches
):
    mini_batches = get_mini_batches

    mar_metrics = MeanAverageRecallAverageOverIOU()

    for mini_batch in mini_batches:
        mar_metrics.update(mini_batch)

    mar_metrics.compute()
    assert mock_update.call_count == len(
        MeanAverageRecallAverageOverIOU.IOU_THRESHOULDS
    ) * len(mini_batches)
    assert mock_compute.call_count == len(
        MeanAverageRecallAverageOverIOU.IOU_THRESHOULDS
    )

    mar_metrics.reset()
    assert mock_reset.call_count == len(
        MeanAverageRecallAverageOverIOU.IOU_THRESHOULDS
    )
