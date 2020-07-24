import shutil
import tempfile
from pathlib import Path

import pandas as pd

import datasetinsights.constants as const
from datasetinsights.data.bbox import BBox2D
from datasetinsights.data.datasets.synthetic import (
    SynDetection2D,
    _get_split,
    read_bounding_box_2d,
)
from datasetinsights.data.simulation import Captures


def test_syn_detection_2d():
    parent_dir = Path(__file__).parent.parent.absolute()
    mock_data_dir = str(parent_dir / "mock_data" / "simrun")
    run_execution_id = "12r46"

    with tempfile.TemporaryDirectory() as tmp_dir:
        dest_path = str(
            Path(tmp_dir) / const.SYNTHETIC_SUBFOLDER / run_execution_id
        )
        shutil.copytree(mock_data_dir, dest_path)
        syn_det_2d = SynDetection2D(
            data_root=tmp_dir, run_execution_id=run_execution_id, def_id=4,
        )

        # From mock data, only one of the capture has 2D bounding box
        # annotations.
        assert len(syn_det_2d) == 1
        assert len(syn_det_2d[0]) == 2


def test_read_bounding_box_2d():
    annotation = pd.DataFrame({
        f"{Captures.VALUES_COLUMN}.instance_id": ["...", "..."],
        f"{Captures.VALUES_COLUMN}.label_id": [27, 30],
        f"{Captures.VALUES_COLUMN}.label_name": ["car", "boy"],
        f"{Captures.VALUES_COLUMN}.x": [30, 40],
        f"{Captures.VALUES_COLUMN}.y": [50, 60],
        f"{Captures.VALUES_COLUMN}.width": [100, 50],
        f"{Captures.VALUES_COLUMN}.height": [80, 60],
    })
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

    assert (
        read_bounding_box_2d(annotation, label_mappings) ==
        [BBox2D(27, 30, 50, 100, 80)]
    )


def test_get_split():
    mock_captures = pd.DataFrame({"id": [i for i in range(10)]})
    actual_train = _get_split(
        split="train", captures=mock_captures, train_percentage=0.6
    )
    actual_val = _get_split(
        split="val", captures=mock_captures, train_percentage=0.6
    )
    expected_train = pd.DataFrame({"id": [3, 8, 0, 9, 6, 7]})
    expected_val = pd.DataFrame({"id": [2, 1, 4, 5]})
    pd.testing.assert_frame_equal(expected_train, actual_train)
    pd.testing.assert_frame_equal(expected_val, actual_val)
