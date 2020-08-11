import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

import datasetinsights.constants as const
from datasetinsights.data.bbox import BBox2D
from datasetinsights.data.exceptions import ChecksumError
from datasetinsights.datasets.synthetic import (
    SynDetection2D,
    _get_split,
    read_bounding_box_2d,
)
from datasetinsights.datasets.synthetic import SynDetection2DDownloader


@patch("datasetinsights.datasets.synthetic._download_captures")
def test_syn_detection_2d(mock_data):
    parent_dir = Path(__file__).parent.parent.absolute()
    mock_data_dir = str(parent_dir / "mock_data" / "simrun")

    with tempfile.TemporaryDirectory() as tmp_dir:
        dest_path = str(Path(tmp_dir) / const.SYNTHETIC_SUBFOLDER)
        shutil.copytree(mock_data_dir, dest_path)
        syn_det_2d = SynDetection2D(data_root=tmp_dir)

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


@patch("datasetinsights.datasets.synthetic.os.path.exists")
@patch("datasetinsights.datasets.synthetic.os.remove")
@patch("datasetinsights.datasets.synthetic.validate_checksum")
@patch("datasetinsights.datasets.synthetic.SynDetection2DDownloader.unzip_file")
def test_synthetic_download_raises_exception(
    mocked_unzip, mocked_validate, mocked_remove, mocked_exists
):
    bad_version = "v_bad"
    version = "v1"
    downloader = SynDetection2DDownloader()
    filename = SynDetection2D.SYNTHETIC_DATASET_TABLES[version].filename
    with tempfile.TemporaryDirectory() as tmp_dir:
        extract_folder = os.path.join(tmp_dir, const.SYNTHETIC_SUBFOLDER)
        dataset_path = os.path.join(extract_folder, filename)

        with pytest.raises(ValueError):
            downloader.download(data_root=tmp_dir, version=bad_version)

    mocked_exists.return_value = True
    mocked_validate.side_effect = ChecksumError()
    downloader.download(tmp_dir, version)
    mocked_remove.assert_called_with(dataset_path)
