"""test dashboard."""
import pandas as pd

from datasetinsights.stats.visualization.object_detection import ScaleFactor


def test_generate_scale_data():
    """test generate scale data."""
    captures = [
        {
            "id": "4521949a- 2a71-4c03-beb0-4f6362676639",
            "sensor": {"scale": 1.0},
        },
        {
            "id": "4b35a47a-3f63-4af3-b0e8-e68cb384ad75",
            "sensor": {"scale": 2.0},
        },
    ]

    captures = pd.DataFrame(captures)
    actual_scale = ScaleFactor.generate_scale_data(captures)
    expected_scale = pd.DataFrame([1.0, 2.0], columns=["scale"])
    pd.testing.assert_frame_equal(expected_scale, actual_scale)
