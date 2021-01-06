from unittest.mock import patch
import numpy as np

from datasetinsights.evaluation_metrics import PrecisionRecallRecords

COMPUTE_RETURN_VALUE = {
    "car": (np.array([1]), np.array([1])),
    "book": (np.array([1]), np.array([1]))
}


@patch("datasetinsights.evaluation_metrics.PrecisionRecallRecords.reset")
@patch("datasetinsights.evaluation_metrics.PrecisionRecallRecords.update")
@patch("datasetinsights.evaluation_metrics.PrecisionRecallRecords.compute")
def test_precision_recall_records(
    mock_compute, mock_update, mock_reset, get_mini_batches
):
    mini_batches = get_mini_batches

    pr_records = PrecisionRecallRecords()
    for mini_batch in mini_batches:
        pr_records.update(mini_batch)

    mock_compute.return_value = COMPUTE_RETURN_VALUE
    res = pr_records.compute()
    assert mock_update.call_count == len(mini_batches)
    assert res == COMPUTE_RETURN_VALUE

    pr_records.reset()
    mock_reset.assert_called()
