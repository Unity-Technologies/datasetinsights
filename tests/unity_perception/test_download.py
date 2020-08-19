import os
import pathlib
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import responses

from datasetinsights.io.download import (
    compute_checksum,
    download_file,
    get_checksum_from_file,
    validate_checksum,
)
from datasetinsights.io.downloader.unity_simulation import (
    Downloader,
    FileType,
    _filter_unsuccessful_attempts,
)
from datasetinsights.io.exceptions import ChecksumError, DownloadError


@pytest.fixture
def downloader():
    parent_dir = pathlib.Path(__file__).parent.parent.absolute()
    manifest_file = str(parent_dir / "mock_data" / "simrun_manifest.csv")

    with tempfile.TemporaryDirectory() as tmp_dir:
        dl = Downloader(manifest_file, tmp_dir)
        yield dl


@responses.activate
def test_download_file_from_url():
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
    source_uri = "https://mock.uri"
    dest_path = "file/path/does/not/matter"
    responses.add(responses.GET, source_uri, status=403)

    with pytest.raises(DownloadError):
        download_file(source_uri, dest_path, False)


def test_download_rows(downloader):
    n_rows = len(downloader.manifest)
    with patch(
        "datasetinsights.io.downloader.unity_simulation.download_file"
    ) as mocked_dl:
        matched_rows = pd.Series(np.zeros(n_rows).astype(bool))
        downloaded = downloader._download_rows(matched_rows)
        assert len(downloaded) == 0
        mocked_dl.assert_not_called()

    with patch(
        "datasetinsights.io.downloader.unity_simulation.download_file"
    ) as mocked_dl:
        matched_rows = pd.Series(np.ones(n_rows).astype(bool))
        downloaded = downloader._download_rows(matched_rows)
        assert len(downloaded) == n_rows
        assert mocked_dl.call_count == n_rows


def test_download_all(downloader):
    n_rows = len(downloader.manifest)
    with patch(
        "datasetinsights.io.downloader.unity_simulation.download_file"
    ) as mocked_dl:
        downloader.download_references()
        downloader.download_captures()
        downloader.download_metrics()
        downloader.download_binary_files()
        assert mocked_dl.call_count == n_rows


def test_filter_unsuccessful_attempts_multiple_ids():
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
    expected_checksum = 123456
    with patch("datasetinsights.io.download._crc32_checksum") as mocked:
        mocked.return_value = expected_checksum
        computed = compute_checksum("filepath/not/important", "CRC32")
        assert computed == expected_checksum

    with pytest.raises(ValueError):
        compute_checksum("filepath/not/important", "UNSUPPORTED_ALGORITHM")


def test_validate_checksum():
    expected_checksum = 123456
    wrong_checksum = 123455
    with patch("datasetinsights.io.download.compute_checksum") as mocked:
        mocked.return_value = wrong_checksum
        with pytest.raises(ChecksumError):
            validate_checksum("filepath/not/important", expected_checksum)


def test_read_checksum_from_local_file():
    # arrange
    with tempfile.NamedTemporaryFile(mode="w+") as tmp:
        tmp.write("123456")
        tmp.flush()
        # act
        checksum = get_checksum_from_file(tmp.name)
        # assert
        assert checksum == 123456


@pytest.mark.parametrize("filepath", ["http://some/path", "https://some/path"])
@patch("datasetinsights.io.download.download_file")
@patch("datasetinsights.io.download.read_checksum_from_txt")
def test_get_checksum_from_http_source(
    mock_read_checksum_from_txt, mock_download_file, filepath
):
    # arrange
    mock_read_checksum_from_txt.return_value = 123456
    # act
    checksum = get_checksum_from_file(filepath)
    # assert
    mock_download_file.assert_called_once()
    assert checksum == 123456


@patch("datasetinsights.io.download.download_file")
@patch("datasetinsights.io.download.read_checksum_from_txt")
def test_get_checksum_from_non_existing_file(
    mock_read_checksum_from_txt, mock_download_file
):
    # arrange
    filepath = "some/wrong/path"
    # assert
    with pytest.raises(ValueError):
        # act
        get_checksum_from_file(filepath)

    # assert
    mock_download_file.assert_not_called()
    mock_read_checksum_from_txt.assert_not_called()
