import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from datasetinsights.data.bbox import BBox2D
from datasetinsights.datasets.synthetic import (
    SYNTHETIC_LOCAL_PATH,
    SynDetection2D,
    _get_split,
    read_bounding_box_2d,
)


@patch("datasetinsights.datasets.synthetic._download_captures")
def test_syn_detection_2d(mock_data):
    parent_dir = Path(__file__).parent.parent.absolute()
    mock_data_dir = str(parent_dir / "mock_data" / "simrun")
    manifest_file = "manifest.csv"

    with tempfile.TemporaryDirectory() as tmp_dir:
        dest_path = str(
            Path(tmp_dir) / SYNTHETIC_LOCAL_PATH / Path(manifest_file).stem
        )
        shutil.copytree(mock_data_dir, dest_path)
        syn_det_2d = SynDetection2D(
            data_root=tmp_dir, manifest_file=manifest_file, def_id=4,
        )

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


def test_get_split():
    mock_catalog = pd.DataFrame({"id": [i for i in range(10)]})
    actual_train = _get_split(
        split="train", catalog=mock_catalog, train_percentage=0.6
    )
    actual_val = _get_split(
        split="val", catalog=mock_catalog, train_percentage=0.6
    )
    expected_train = pd.DataFrame({"id": [3, 8, 0, 9, 6, 7]})
    expected_val = pd.DataFrame({"id": [2, 1, 4, 5]})
    pd.testing.assert_frame_equal(expected_train, actual_train)
    pd.testing.assert_frame_equal(expected_val, actual_val)
