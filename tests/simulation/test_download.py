import os
import pathlib
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import responses

from datasetinsights.data.download import download_file
from datasetinsights.data.simulation.download import (
    Downloader,
    DownloadError,
    _filter_unsuccessful_attempts,
)
from datasetinsights.data.simulation.tables import FileType


@pytest.fixture
def downloader():
    parent_dir = pathlib.Path(__file__).parent.parent.absolute()
    manifest_file = str(parent_dir / "mock_data" / "simrun_manifest.csv")

    with tempfile.TemporaryDirectory() as tmp_dir:
        dl = Downloader(manifest_file, tmp_dir)
        yield dl


@responses.activate
def test_download_file():
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
    with patch("datasetinsights.data.download.download_file") as mocked_dl:
        matched_rows = pd.Series(np.zeros(n_rows).astype(bool))
        downloaded = downloader._download_rows(matched_rows)
        assert len(downloaded) == 0
        mocked_dl.assert_not_called()

    with patch("datasetinsights.data.download.download_file") as mocked_dl:
        matched_rows = pd.Series(np.ones(n_rows).astype(bool))
        downloaded = downloader._download_rows(matched_rows)
        assert len(downloaded) == n_rows
        assert mocked_dl.call_count == n_rows


def test_download_all(downloader):
    n_rows = len(downloader.manifest)
    with patch("datasetinsights.data.download.download_file") as mocked_dl:
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
