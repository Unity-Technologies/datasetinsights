"""test download."""
import os
import pathlib
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import responses

from datasetinsights.data.download import (
    compute_checksum,
    download_file,
    validate_checksum,
)
from datasetinsights.data.exceptions import ChecksumError, DownloadError
from datasetinsights.datasets.simulation import (
    Downloader,
    FileType,
    _filter_unsuccessful_attempts,
)


@pytest.fixture
def downloader():
    """downloader."""
    parent_dir = pathlib.Path(__file__).parent.parent.absolute()
    manifest_file = str(parent_dir / "mock_data" / "simrun_manifest.csv")

    with tempfile.TemporaryDirectory() as tmp_dir:
        dl = Downloader(manifest_file, tmp_dir)
        yield dl


@responses.activate
def test_download_file_from_url():
    """test download file from url."""
    source_uri = "https://mock.uri"
    body = b"some test string here"
    responses.add(
        responses.GET, source_uri, body=body, content_type="text/plain"
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        dest_path = os.path.join(tmp_dir, "test.txt")
        download_file(source_uri, dest_path, False)

        assert os.path.exists(dest_path)
        assert open(dest_path, "rb").read() == body


def test_download_bad_request():
    """test download bad request."""
    source_uri = "https://mock.uri"
    dest_path = "file/path/does/not/matter"
    responses.add(responses.GET, source_uri, status=403)

    with pytest.raises(DownloadError):
        download_file(source_uri, dest_path, False)


def test_download_rows(downloader):
    """test download rows."""
    n_rows = len(downloader.manifest)
    with patch(
        "datasetinsights.datasets.simulation.download.download_file"
    ) as mocked_dl:
        matched_rows = pd.Series(np.zeros(n_rows).astype(bool))
        downloaded = downloader._download_rows(matched_rows)
        assert len(downloaded) == 0
        mocked_dl.assert_not_called()

    with patch(
        "datasetinsights.datasets.simulation.download.download_file"
    ) as mocked_dl:
        matched_rows = pd.Series(np.ones(n_rows).astype(bool))
        downloaded = downloader._download_rows(matched_rows)
        assert len(downloaded) == n_rows
        assert mocked_dl.call_count == n_rows


def test_download_all(downloader):
    """test download all."""
    n_rows = len(downloader.manifest)
    with patch(
        "datasetinsights.datasets.simulation.download.download_file"
    ) as mocked_dl:
        downloader.download_references()
        downloader.download_captures()
        downloader.download_metrics()
        downloader.download_binary_files()
        assert mocked_dl.call_count == n_rows


def test_filter_unsuccessful_attempts_multiple_ids():
    """test filter unsuccessful attempts multiple ids."""
    manifest_df = pd.DataFrame(
        {
            "run_execution_id": ["a"] * 8,
            "attempt_id": [0, 1, 0, 0, 0, 1, 2, 3],
            "app_param_id": [47, 47, 22, 22, 50, 50, 50, 50],
            "instance_id": [0, 0, 1, 1, 2, 2, 2, 2],
        }
    )
    expected_result = pd.DataFrame(
        {
            "run_execution_id": ["a"] * 4,
            "attempt_id": [1, 0, 0, 3],
            "app_param_id": [47, 22, 22, 50],
            "instance_id": [0, 1, 1, 2],
        }
    )
    actual_result = _filter_unsuccessful_attempts(manifest_df)
    pd.testing.assert_frame_equal(expected_result, actual_result)


def test_filter_unsuccessful_attempts_single_attempt_id():
    """test filter unsuccessful attempts single attempt id."""
    manifest_df = pd.DataFrame(
        {
            "run_execution_id": ["a", "a"],
            "attempt_id": [0, 0],
            "app_param_id": [47, 52],
            "instance_id": [0, 0],
        }
    )
    expected_result = pd.DataFrame(
        {
            "run_execution_id": ["a", "a"],
            "attempt_id": [0, 0],
            "app_param_id": [47, 52],
            "instance_id": [0, 0],
        }
    )
    actual_result = _filter_unsuccessful_attempts(manifest_df)
    pd.testing.assert_frame_equal(expected_result, actual_result)


def test_match_filetypes():
    """test match filetypes."""
    manifest = pd.DataFrame(
        {
            "file_name": [
                "abc/dfv.png",
                "Dataset/annotation_definitions.json",
                "Dataset/metrics_04323423.json",
                "Dataset/metric_definitions.json",
                "Dataset/sensors.json",
                "Dataset/captures_000123153.json",
                "Dataset/egos.json",
                "segmentation/image_9013.png",
                "lidar/points_9013.pcd",
            ]
        }
    )
    expected_filetypes = [
        FileType.BINARY,
        FileType.REFERENCE,
        FileType.METRIC,
        FileType.REFERENCE,
        FileType.REFERENCE,
        FileType.CAPTURE,
        FileType.REFERENCE,
        FileType.BINARY,
        FileType.BINARY,
    ]

    assert Downloader.match_filetypes(manifest) == expected_filetypes


def test_compute_checksum():
    """test compute checksum."""
    expected_checksum = 123456
    with patch("datasetinsights.data.download._crc32_checksum") as mocked:
        mocked.return_value = expected_checksum
        computed = compute_checksum("filepath/not/important", "CRC32")
        assert computed == expected_checksum

    with pytest.raises(ValueError):
        compute_checksum("filepath/not/important", "UNSUPPORTED_ALGORITHM")


def test_validate_checksum():
    """test validate checksum."""
    expected_checksum = 123456
    wrong_checksum = 123455
    with patch("datasetinsights.data.download.compute_checksum") as mocked:
        mocked.return_value = wrong_checksum
        with pytest.raises(ChecksumError):
            validate_checksum("filepath/not/important", expected_checksum)
