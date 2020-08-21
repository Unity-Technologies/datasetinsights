from unittest.mock import MagicMock, patch

from datasetinsights.io.compression import (
    GZipCompression,
    TarFileCompression,
    ZipFileCompression,
    compression_factory,
)


@patch("datasetinsights.io.compression._get_file_extension_from_filepath")
def test_compression_factory_returns_zipfile_compression(
    mocked_get_file_extension,
):
    mocked_get_file_extension.return_value = "zip"
    assert compression_factory(filepath=MagicMock()) == ZipFileCompression
    mocked_get_file_extension.assert_called_once()


@patch("datasetinsights.io.compression._get_file_extension_from_filepath")
def test_compression_factory_returns_tarfile_compression(
    mocked_get_file_extension,
):
    mocked_get_file_extension.return_value = "tar"
    assert compression_factory(filepath=MagicMock()) == TarFileCompression
    mocked_get_file_extension.assert_called_once()


@patch("datasetinsights.io.compression._get_file_extension_from_filepath")
def test_compression_factory_returns_gzip_compression(
    mocked_get_file_extension,
):
    mocked_get_file_extension.return_value = "gz"
    assert compression_factory(filepath=MagicMock()) == GZipCompression
    mocked_get_file_extension.assert_called_once()
