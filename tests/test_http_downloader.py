import tempfile
from unittest.mock import patch

import pytest

from datasetinsights.io.downloader.http_downloader import HTTPDownloader
from datasetinsights.io.exceptions import ChecksumError


def test_read_checksum_from_local_file():
    # arrange
    with tempfile.NamedTemporaryFile(mode="w+") as tmp:
        tmp.write("123456")
        tmp.flush()
        # act
        checksum = HTTPDownloader.get_checksum_from_file(tmp.name)
        # assert
        assert checksum == 123456


@pytest.mark.parametrize("filepath", ["http://some/path", "https://some/path"])
@patch("datasetinsights.io.downloader.http_downloader.download_file")
@patch.object(HTTPDownloader, "read_checksum_from_txt")
def test_get_checksum_from_http_source(
    mock_read_checksum_from_txt, mock_download_file, filepath
):
    # arrange
    mock_read_checksum_from_txt.return_value = 123456
    # act
    checksum = HTTPDownloader.get_checksum_from_file(filepath)
    # assert
    mock_download_file.assert_called_once()
    assert checksum == 123456


@patch("datasetinsights.io.downloader.http_downloader.download_file")
@patch.object(HTTPDownloader, "read_checksum_from_txt")
def test_get_checksum_from_non_existing_file(
    mock_read_checksum_from_txt, mock_download_file
):
    # arrange
    filepath = "some/wrong/path"
    # assert
    with pytest.raises(ValueError):
        # act
        HTTPDownloader.get_checksum_from_file(filepath)

    # assert
    mock_download_file.assert_not_called()
    mock_read_checksum_from_txt.assert_not_called()


@pytest.mark.parametrize(
    "source_uri",
    ["http://some/path", "https://some/path", "some/checksum_file.txt"],
)
@patch("datasetinsights.io.downloader.http_downloader.download_file")
@patch.object(HTTPDownloader, "unzip_file")
def test_download_without_checksum(mock_unzip, mock_download_file, source_uri):
    # arrange
    output = "/some/path/"
    dataset_path = output + "dataset.zip"
    downloader = HTTPDownloader()

    # act
    downloader.download(source_uri=source_uri, output=output)

    # assert
    mock_download_file.assert_called_once()
    mock_unzip.assert_called_once_with(dataset_path, output)


@pytest.mark.parametrize(
    "source_uri", ["http://some/path", "https://some/path"]
)
@pytest.mark.parametrize(
    "checksum_file",
    ["http://some/path", "https://some/path", "/some/checksum_file.txt"],
)
@patch("datasetinsights.io.downloader.http_downloader.download_file")
@patch("datasetinsights.io.downloader.http_downloader.validate_checksum")
@patch.object(HTTPDownloader, "unzip_file")
@patch.object(HTTPDownloader, "get_checksum_from_file")
def test_download_with_checksum(
    mock_get_checksum_from_file,
    mock_unzip,
    mock_validate_check_sum,
    mock_download_file,
    source_uri,
    checksum_file,
):
    # arrange"
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


@pytest.mark.parametrize(
    "source_uri", ["http://some/path", "https://some/path"]
)
@pytest.mark.parametrize(
    "checksum_file",
    ["http://some/path", "https://some/path", "/some/checksum_file.txt"],
)
@patch("os.remove")
@patch("datasetinsights.io.downloader.http_downloader.download_file")
@patch("datasetinsights.io.downloader.http_downloader.validate_checksum")
@patch.object(HTTPDownloader, "unzip_file")
@patch.object(HTTPDownloader, "get_checksum_from_file")
def test_download_with_wrong_checksum(
    mock_get_checksum_from_file,
    mock_unzip_file,
    mock_validate_checksum,
    mock_download_file,
    mock_remove,
    source_uri,
    checksum_file,
):
    # arrange
    mock_validate_checksum.side_effect = ChecksumError
    output = "/some/path"
    downloader = HTTPDownloader()

    # act
    with pytest.raises(ChecksumError):
        downloader.download(
            source_uri=source_uri, output=output, checksum_file=checksum_file
        )

    # assert
    mock_get_checksum_from_file.assert_called_once()
    mock_download_file.asert_called_once()
    mock_remove.assert_called_once()
    mock_unzip_file.assert_not_called()
