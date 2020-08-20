from unittest.mock import patch

import pytest

from datasetinsights.io.downloader.http_downloader import HTTPDownloader
from datasetinsights.io.exceptions import ChecksumError


@patch("datasetinsights.io.downloader.http_downloader.download_file")
@patch.object(HTTPDownloader, "unzip_file")
def test_download_without_checksum(mock_unzip, mock_download_file):
    # arrange
    source_uri = "http://some/path"
    output = "/some/path/"
    dataset_path = output + "dataset.zip"
    downloader = HTTPDownloader()

    # act
    downloader.download(source_uri=source_uri, output=output)

    # assert
    mock_download_file.assert_called_once()
    mock_unzip.assert_called_once_with(dataset_path, output)


@patch("datasetinsights.io.downloader.http_downloader.download_file")
@patch("datasetinsights.io.downloader.http_downloader.validate_checksum")
@patch("datasetinsights.io.downloader.http_downloader.get_checksum_from_file")
@patch.object(HTTPDownloader, "unzip_file")
def test_download_with_checksum(
    mock_unzip,
    mock_get_checksum_from_file,
    mock_validate_check_sum,
    mock_download_file,
):
    # arrange
    source_uri = "http://some/path"
    checksum_file = "/some/checksum_file.txt"
    output = "/some/path/"
    dataset_path = output + "dataset.zip"
    downloader = HTTPDownloader()

    # act
    downloader.download(
        source_uri=source_uri, output=output, checksum_file=checksum_file
    )

    # assert
    mock_download_file.assert_called_once()
    mock_unzip.assert_called_once_with(dataset_path, output)
    mock_get_checksum_from_file.assert_called_once()
    mock_validate_check_sum.assert_called_once()


@patch("os.remove")
@patch("datasetinsights.io.downloader.http_downloader.download_file")
@patch("datasetinsights.io.downloader.http_downloader.validate_checksum")
@patch("datasetinsights.io.downloader.http_downloader.get_checksum_from_file")
@patch.object(HTTPDownloader, "unzip_file")
def test_download_with_wrong_checksum(
    mock_unzip,
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
    downloader = HTTPDownloader()

    # act
    with pytest.raises(ChecksumError):
        downloader.download(
            source_uri=source_uri, output=output, checksum_file=checksum_file
        )

    # assert
    mock_get_checksum_from_file.assert_called_once()
    mock_download_file.assert_called_once()
    mock_remove.assert_called_once()
    mock_unzip.assert_not_called()
