import os
import tempfile
from pathlib import Path
from pytest import raises

from unittest.mock import patch

from datasetinsights.datasets.coco import CocoDetection
from datasetinsights.datasets.exceptions import DatasetNotFoundError


def test__is_dataset_files_present():
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "coco.json"), "x"):
            with open(os.path.join(tmp, "coco.jpg"), "x"):
                assert CocoDetection._is_dataset_files_present(tmp)

    with tempfile.TemporaryDirectory() as tmp:
        assert not CocoDetection._is_dataset_files_present(tmp)


@patch("datasetinsights.datasets.CocoDetection._unarchive_data")
def test__preprocess_dataset(mock_unarchive):
    tmp_dir = tempfile.TemporaryDirectory()
    tmp_name = tmp_dir.name
    split = "train"

    # test no dataset found
    with raises(DatasetNotFoundError):
        CocoDetection._preprocess_dataset(tmp_name, split)

    # test dataset already exists
    with open(os.path.join(tmp_name, "coco.json"), "x"):
        with open(os.path.join(tmp_name, "coco.jpg"), "x"):
            return_value = CocoDetection._preprocess_dataset(tmp_name, split)
            assert return_value == tmp_name

    # test whether it can unarchive data
    archive_img_file = Path(tmp_name) / f"{split}2017.zip"
    archive_ann_file = Path(tmp_name) / "annotations_trainval2017.zip"
    with open(archive_img_file, "x"):
        with open(archive_ann_file, "x"):
            CocoDetection._preprocess_dataset(tmp_name, split)
            assert mock_unarchive.call_count == 2

    tmp_dir.cleanup()
