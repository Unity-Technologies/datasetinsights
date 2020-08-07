import pandas as pd

from datasetinsights.stats.statistics import RenderedObjectInfo


def test_read_filtered_metrics():
    metrics = pd.DataFrame(
        {
            "capture_id": [
                "",
                "1231",
                "1231",
                "1231",
                "2324",
                "323523",
                "323523",
            ],
            "label_id": [0, 1, 2, 3, 1, 2, 3],
            "label_name": ["", "car", "bike", "child", "car", "bike", "child"],
            "value": [0, 2, 3, 1, 1, 1, 4],
        }
    )
    mappings = {1: "car", 2: "bike", 3: "child"}
    expected = pd.DataFrame(
        {
            "capture_id": ["1231", "1231", "1231", "2324", "323523", "323523"],
            "label_id": [1, 2, 3, 1, 2, 3],
            "label_name": ["car", "bike", "child", "car", "bike", "child"],
            "value": [2, 3, 1, 1, 1, 4],
        }
    )

    agg = RenderedObjectInfo._read_filtered_metrics(metrics, mappings)
    agg = agg.reset_index(drop=True)
    pd.testing.assert_frame_equal(agg, expected, check_like=True)
