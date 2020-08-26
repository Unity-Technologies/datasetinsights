from unittest.mock import patch

import pytest

from datasetinsights.io.downloader.http_downloader import HTTPDatasetDownloader
from datasetinsights.io.exceptions import ChecksumError


@patch(
    "datasetinsights.io.downloader.http_downloader."
    "download_dataset_from_http_url"
)
def test_download_without_checksum(mock_download_file):
    # arrange
    source_uri = "http://some/path"
    output = "/some/path/"
    downloader = HTTPDatasetDownloader()

    # act
    downloader.download(source_uri=source_uri, output=output)

    # assert
    mock_download_file.assert_called_once()


@patch(
    "datasetinsights.io.downloader.http_downloader."
    "download_dataset_from_http_url"
)
@patch("datasetinsights.io.downloader.http_downloader.validate_checksum")
@patch("datasetinsights.io.downloader.http_downloader.get_checksum_from_file")
def test_download_with_checksum(
    mock_get_checksum_from_file, mock_validate_check_sum, mock_download_file,
):
    # arrange
    source_uri = "http://some/path"
    checksum_file = "/some/checksum_file.txt"
    output = "/some/path/"
    downloader = HTTPDatasetDownloader()

    # act
    downloader.download(
        source_uri=source_uri, output=output, checksum_file=checksum_file
    )

    # assert
    mock_download_file.assert_called_once()
    mock_get_checksum_from_file.assert_called_once()
    mock_validate_check_sum.assert_called_once()


@patch("os.remove")
@patch(
    "datasetinsights.io.downloader.http_downloader."
    "download_dataset_from_http_url"
)
@patch("datasetinsights.io.downloader.http_downloader.validate_checksum")
@patch("datasetinsights.io.downloader.http_downloader.get_checksum_from_file")
def test_download_with_wrong_checksum(
    mock_get_checksum_from_file,
    mock_validate_checksum,
    mock_download_file,
    mock_remove,
):
    # arrange
    mock_validate_checksum.side_effect = ChecksumError
    output = "/some/path"
    source_uri = "http://some/path"
    checksum_file = "/some/checksum_file.txt"
    downloader = HTTPDatasetDownloader()

    # act
    with pytest.raises(ChecksumError):
        downloader.download(
            source_uri=source_uri, output=output, checksum_file=checksum_file
        )

    # assert
    mock_get_checksum_from_file.assert_called_once()
    mock_download_file.assert_called_once()
    mock_remove.assert_called_once()
