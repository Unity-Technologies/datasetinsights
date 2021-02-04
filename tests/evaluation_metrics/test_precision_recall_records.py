from pytest import approx

from datasetinsights.evaluation_metrics import PrecisionRecallRecords


def test_precision_recall_records(get_mini_batches):
    mini_batches = get_mini_batches

    pr_records = PrecisionRecallRecords()
    expected_precision = [1, 0.5, 0.33333, 0.5]
    expected_recall = [0.33333, 0.33333, 0.33333, 0.66667]
    for mini_batch in mini_batches:
        pr_records.update(mini_batch)

    res = pr_records.compute()
    precision_ped, recall_ped = res["pedestrian"]
    assert len(precision_ped) == len(expected_precision)
    assert len(recall_ped) == len(expected_recall)
    for i in range(len(precision_ped)):
        assert (
            approx(precision_ped[i], rel=1e-4) == expected_precision[i]
        )
    for i in range(len(recall_ped)):
        assert (
            approx(recall_ped[i], rel=1e-4) == expected_recall[i]
        )
