import tempfile
from unittest.mock import patch

import pytest

from datasetinsights.io.downloader.http_downloader import HTTPDownloader
from datasetinsights.io.exceptions import ChecksumError


def test_read_checksum_from_txt():
    with tempfile.NamedTemporaryFile(mode="w+") as tmp:
        tmp.write("123456")
        tmp.read()
        assert HTTPDownloader.get_checksum_from_file(tmp.name) == 123456


@patch("datasetinsights.io.downloader.http_downloader.download_file")
@patch.object(HTTPDownloader, "read_checksum_from_txt")
def test_get_checksum_from_file(
    mock_read_checksum_from_txt, mock_download_file
):
    filepath_1 = "http://some/path"
    filepath_2 = "https://some/path"
    filepath_3 = "some/wrong/path"

    mock_read_checksum_from_txt.return_value = 123456

    HTTPDownloader.get_checksum_from_file(filepath_1)
    mock_download_file.assert_called_once()
    mock_read_checksum_from_txt.assert_called_once()

    assert HTTPDownloader.get_checksum_from_file(filepath_2) == 123456

    with pytest.raises(ValueError):
        HTTPDownloader.get_checksum_from_file(filepath_3)

    with tempfile.NamedTemporaryFile() as tmp:
        assert HTTPDownloader.get_checksum_from_file(tmp.name) == 123456


@patch("os.remove")
@patch("datasetinsights.io.downloader.http_downloader.download_file")
@patch("datasetinsights.io.downloader.http_downloader.validate_checksum")
@patch.object(HTTPDownloader, "get_checksum_from_file")
def test_download(
    mock_get_checksum_from_file,
    mock_validate_checksum,
    mock_download_file,
    mock_remove,
):
    mock_validate_checksum.side_effect = ChecksumError
    with pytest.raises(ChecksumError):
        downloader = HTTPDownloader()
        downloader.download(
            source_uri="http://some/path",
            output="/some/path",
            checksum_file="https://some/path",
        )
        mock_get_checksum_from_file.assert_called_once()
        mock_download_file.assert_called_once()
        mock_remove.assert_called_once()
