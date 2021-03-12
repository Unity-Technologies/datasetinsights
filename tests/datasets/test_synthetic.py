from datasetinsights.datasets.synthetic import read_bounding_box_2d
from datasetinsights.io.bbox import BBox2D


def test_read_bounding_box_2d():
    annotation = [
        {
            "instance_id": "...",
            "label_id": 27,
            "label_name": "car",
            "x": 30,
            "y": 50,
            "width": 100,
            "height": 100,
        }
    ]
    definition = {
        "id": 1243,
        "name": "...",
        "description": "...",
        "format": "JSON",
        "spec": [{"label_id": 27, "label_name": "car"}],
    }
    label_mappings = {
        m["label_id"]: m["label_name"] for m in definition["spec"]
    }
    bbox = read_bounding_box_2d(annotation, label_mappings)

    assert bbox == [BBox2D(27, 30, 50, 100, 100)]
