from unittest.mock import patch

from pytest import approx

from datasetinsights.evaluation_metrics import (
    AveragePrecisionIOU50,
    AveragePrecision,
    MeanAveragePrecisionAverageOverIOU,
    MeanAveragePrecisionIOU50,
)


def test_average_precision_2d_bbox(get_mini_batches):
    mini_batches = get_mini_batches

    # test iou threshold = 0.5
    ap_metrics = AveragePrecision(iou_threshold=0.5)
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


@patch("datasetinsights.evaluation_metrics.AveragePrecision.reset")
@patch("datasetinsights.evaluation_metrics.AveragePrecision.update")
@patch("datasetinsights.evaluation_metrics.AveragePrecision.compute")
def test_mean_average_precision_average_over_iou(
    mock_compute, mock_update, mock_reset, get_mini_batches
):
    mini_batches = get_mini_batches

    map_metrics = MeanAveragePrecisionAverageOverIOU()

    for mini_batch in mini_batches:
        map_metrics.update(mini_batch)

    map_metrics.compute()
    assert mock_update.call_count == len(
        MeanAveragePrecisionAverageOverIOU.IOU_THRESHOULDS
    ) * len(mini_batches)
    assert mock_compute.call_count == len(
        MeanAveragePrecisionAverageOverIOU.IOU_THRESHOULDS
    )

    map_metrics.reset()
    mock_reset.call_count == len(
        MeanAveragePrecisionAverageOverIOU.IOU_THRESHOULDS
    )


@patch("datasetinsights.evaluation_metrics.AveragePrecision.reset")
@patch("datasetinsights.evaluation_metrics.AveragePrecision.update")
@patch("datasetinsights.evaluation_metrics.AveragePrecision.compute")
def test_average_precision_IOU50(
    mock_compute, mock_update, mock_reset, get_mini_batches
):
    mini_batches = get_mini_batches

    map_metrics = AveragePrecisionIOU50()
    for mini_batch in mini_batches:
        map_metrics.update(mini_batch)

    map_metrics.compute()
    assert mock_update.call_count == len(mini_batches)
    mock_compute.assert_called()

    map_metrics.reset()
    mock_reset.assert_called()


@patch("datasetinsights.evaluation_metrics.AveragePrecision.reset")
@patch("datasetinsights.evaluation_metrics.AveragePrecision.update")
@patch("datasetinsights.evaluation_metrics.AveragePrecision.compute")
def test_mean_average_precision_IOU50(
    mock_compute, mock_update, mock_reset, get_mini_batches
):
    mini_batches = get_mini_batches

    map_metrics = MeanAveragePrecisionIOU50()
    for mini_batch in mini_batches:
        map_metrics.update(mini_batch)

    map_metrics.compute()
    assert mock_update.call_count == len(mini_batches)
    mock_compute.assert_called()

    map_metrics.reset()
    mock_reset.assert_called()
