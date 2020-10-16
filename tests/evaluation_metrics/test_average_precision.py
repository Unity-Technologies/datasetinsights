from datasetinsights.evaluation_metrics import average_precision
from datasetinsights.evaluation_metrics.average_precision import BBox3D


def _make_box(x, y, z, sample_token, confidence=None):
    return BBox3D(
        translation=(x, y, z),
        size=(1, 1, 1),
        velocity=(0, 0),
        score=confidence,
        label="truck",
        sample_token=sample_token,
    )


def test_ap_and_map():
    ap = average_precision.AveragePrecision()
    boxes = {
        "image1": (
            [
                _make_box(0, 0, 0, confidence=0.95, sample_token="image1"),
                _make_box(
                    100000, 100, 100, confidence=0.9, sample_token="image1"
                ),
            ],
            [_make_box(0, 0, 0.1, sample_token="image1")],
        )
    }
    ap.update(boxes=boxes)
    actual_ap = ap.compute()
    expected_ap = {
        "car": 0.0,
        "truck": 0.9938271604938275,
        "bus": 0.0,
        "trailer": 0.0,
        "construction_vehicle": 0.0,
        "pedestrian": 0.0,
        "motorcycle": 0.0,
        "bicycle": 0.0,
        "traffic_cone": 0.0,
        "barrier": 0.0,
    }
    assert expected_ap == actual_ap
