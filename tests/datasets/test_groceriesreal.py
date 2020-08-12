import os
import tempfile
from unittest.mock import patch

import pytest

from datasetinsights.datasets import GroceriesReal
from datasetinsights.io.exceptions import ChecksumError, DownloadError


@patch("datasetinsights.datasets.groceries_real.os.remove")
@patch("datasetinsights.datasets.groceries_real.download_file")
@patch("datasetinsights.datasets.groceries_real.validate_checksum")
def test_groceriesreal_download_http(
    mocked_validate, mocked_download, mocked_remove
):
    dummy_uri = "https://mock.uri"
    version = "v3"
    with tempfile.TemporaryDirectory() as tmp_dir:
        mocked_download.return_value = tmp_dir
        expected_checksum = GroceriesReal.GROCERIES_REAL_DATASET_TABLES[
            version
        ].checksum
        GroceriesReal._download_http(dummy_uri, tmp_dir, version)
        mocked_download.assert_called_with(
            source_uri=dummy_uri, dest_path=tmp_dir
        )
        mocked_validate.assert_called_with(tmp_dir, expected_checksum)

        mocked_download.side_effect = DownloadError()
        with pytest.raises(DownloadError):
            GroceriesReal._download_http(dummy_uri, tmp_dir, version)

        mocked_download.side_effect = None
        mocked_validate.side_effect = ChecksumError()
        with pytest.raises(ChecksumError):
            GroceriesReal._download_http(dummy_uri, tmp_dir, version)
            mocked_remove.assert_called()


@patch("datasetinsights.datasets.groceries_real.os.path.exists")
@patch("datasetinsights.datasets.groceries_real.os.remove")
@patch("datasetinsights.datasets.groceries_real.validate_checksum")
@patch("datasetinsights.datasets.groceries_real.GroceriesReal._extract_file")
@patch("datasetinsights.datasets.groceries_real.GroceriesReal._download_http")
def test_groceriesreal_download(
    mocked_download,
    mocked_extract,
    mocked_validate,
    mocked_remove,
    mocked_exists,
):
    version = "v3"
    source_uri = GroceriesReal.GROCERIES_REAL_DATASET_TABLES[version].source_uri
    expected_checksum = GroceriesReal.GROCERIES_REAL_DATASET_TABLES[
        version
    ].checksum
    with tempfile.TemporaryDirectory() as tmp_dir:
        extract_folder = os.path.join(tmp_dir, GroceriesReal.LOCAL_PATH)
        dest_path = os.path.join(
            tmp_dir, GroceriesReal.LOCAL_PATH, f"{version}.zip"
        )
        mocked_exists.return_value = True
        GroceriesReal.download(tmp_dir, version)
        mocked_validate.assert_called_with(dest_path, expected_checksum)
        mocked_extract.assert_called_with(dest_path, extract_folder)

        mocked_exists.return_value = False
        GroceriesReal.download(tmp_dir, version)
        mocked_download.assert_called_with(source_uri, dest_path, version)
        mocked_extract.assert_called_with(dest_path, extract_folder)


@patch("datasetinsights.datasets.groceries_real.os.path.exists")
@patch("datasetinsights.datasets.groceries_real.os.remove")
@patch("datasetinsights.datasets.groceries_real.validate_checksum")
@patch("datasetinsights.datasets.groceries_real.GroceriesReal._extract_file")
@patch("datasetinsights.datasets.groceries_real.GroceriesReal._download_http")
def test_groceriesreal_download_raises(
    mocked_download,
    mocked_extract,
    mocked_validate,
    mocked_remove,
    mocked_exists,
):
    version = "v3"
    bad_version = "v_bad"
    source_uri = GroceriesReal.GROCERIES_REAL_DATASET_TABLES[version].source_uri
    with tempfile.TemporaryDirectory() as tmp_dir:
        dest_path = os.path.join(
            tmp_dir, GroceriesReal.LOCAL_PATH, f"{version}.zip"
        )
        extract_folder = os.path.join(tmp_dir, GroceriesReal.LOCAL_PATH)
        with pytest.raises(ValueError):
            GroceriesReal.download(tmp_dir, bad_version)

        mocked_exists.return_value = True
        mocked_validate.side_effect = ChecksumError()
        GroceriesReal.download(tmp_dir, version)
        mocked_remove.assert_called_with(dest_path)
        mocked_extract.assert_called_with(dest_path, extract_folder)

        mocked_exists.return_value = False
        mocked_validate.side_effect = None
        GroceriesReal.download(tmp_dir, version)
        mocked_extract.assert_called_with(dest_path, extract_folder)
        mocked_download.assert_called_with(source_uri, dest_path, version)
