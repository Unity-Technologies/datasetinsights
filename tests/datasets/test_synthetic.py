import os
import tempfile
from pathlib import Path

from datasetinsights.datasets.synthetic import (
    SynDetection2D,
    read_bounding_box_2d,
)
from datasetinsights.io.bbox import BBox2D


def test_syn_detection_2d():
    parent_dir = Path(__file__).parent.parent.absolute()
    mock_data_dir = str(parent_dir / "mock_data" / "simrun")
    syn_det_2d = SynDetection2D(data_path=mock_data_dir)

    # From mock data, only one of the capture has 2D bounding box
    # annotations.
    assert len(syn_det_2d) == 1
    assert len(syn_det_2d[0]) == 2


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


def test_is_dataset_files_present_returns_true():
    with tempfile.TemporaryDirectory() as tmp:
        temp_dir = os.path.join(tmp, "temp_name")
        os.mkdir(temp_dir)
        with open(os.path.join(temp_dir, "annotation.json"), "x"):
            assert SynDetection2D.is_dataset_files_present(tmp)


def test_is_dataset_files_present_returns_false():
    with tempfile.TemporaryDirectory() as tmp:
        temp_dir = os.path.join(tmp, "temp_name")
        os.mkdir(temp_dir)
        assert not SynDetection2D.is_dataset_files_present(tmp)
