from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from datasetinsights.io.downloader import GCSDatasetDownloader
from datasetinsights.io.exceptions import ChecksumError
from datasetinsights.io.gcs import GCSClient

bucket_name = "fake_bucket"
local_path = "path/to/local"
md5_hash = "abc=="
md5_hash_hex = "12345"
file_name = "/data.zip"
base_key = "path/to/object"
base_url = "gs://fake_bucket/path/to/object"


@patch("datasetinsights.io.gcs.GCSClient._upload_file")
@patch("datasetinsights.io.gcs.isdir")
def test_gcs_client_upload_file_bucket_key(mock_isdir, mock_upload_file):
    localfile = local_path + file_name
    mocked_gcs_client = MagicMock()
    mock_isdir.return_value = False
    with patch(
        "datasetinsights.io.gcs.Client",
        MagicMock(return_value=mocked_gcs_client),
    ):
        client = GCSClient()
        client.upload(local_path=localfile, bucket=bucket_name, key=base_key)
        mock_upload_file.assert_called_with(
            bucket=mocked_gcs_client.get_bucket(),
            key=base_key,
            local_path=localfile,
        )


@patch("datasetinsights.io.gcs.GCSClient._upload_file")
@patch("datasetinsights.io.gcs.isdir")
def test_gcs_client_upload_file_url(mock_isdir, mock_upload_file):
    localfile = local_path + file_name
    mocked_gcs_client = MagicMock()
    mock_isdir.return_value = False
    url = base_url + file_name
    with patch(
        "datasetinsights.io.gcs.Client",
        MagicMock(return_value=mocked_gcs_client),
    ):
        client = GCSClient()
        client.upload(local_path=localfile, url=url)
        mock_upload_file.assert_called_with(
            bucket=mocked_gcs_client.get_bucket(),
            key=base_key + file_name,
            local_path=localfile,
        )


@patch("datasetinsights.io.gcs.GCSClient._upload_folder")
@patch("datasetinsights.io.gcs.isdir")
def test_gcs_client_upload_folder_bucket_key(mock_isdir, mock_upload_folder):
    mocked_gcs_client = MagicMock()
    mock_isdir.return_value = True
    with patch(
        "datasetinsights.io.gcs.Client",
        MagicMock(return_value=mocked_gcs_client),
    ):
        client = GCSClient()
        client.upload(
            local_path=local_path, bucket=bucket_name, key=base_key, pattern="*"
        )
        mock_upload_folder.assert_called_with(
            bucket=mocked_gcs_client.get_bucket(),
            key=base_key,
            local_path=local_path,
            pattern="*",
        )


@patch("datasetinsights.io.gcs.GCSClient._upload_folder")
@patch("datasetinsights.io.gcs.isdir")
def test_gcs_client_upload_folder_url(mock_isdir, mock_upload_folder):
    mocked_gcs_client = MagicMock()
    mock_isdir.return_value = True
    url = base_url
    with patch(
        "datasetinsights.io.gcs.Client",
        MagicMock(return_value=mocked_gcs_client),
    ):
        client = GCSClient()
        client.upload(local_path=local_path, url=url, pattern="*")
        mock_upload_folder.assert_called_with(
            bucket=mocked_gcs_client.get_bucket(),
            key=base_key,
            local_path=local_path,
            pattern="*",
        )


@patch("datasetinsights.io.gcs.GCSClient._is_file")
@patch("datasetinsights.io.gcs.GCSClient._download_file")
def test_gcs_client_download_file_bucket_key(mock_download_file, mock_is_file):
    mocked_gcs_client = MagicMock()
    mock_is_file.return_value = True
    object_key = base_key + file_name
    with patch(
        "datasetinsights.io.gcs.Client",
        MagicMock(return_value=mocked_gcs_client),
    ):
        client = GCSClient()
        client.download(
            local_path=local_path, bucket=bucket_name, key=object_key
        )
        mock_download_file.assert_called_with(
            mocked_gcs_client.get_bucket(), object_key, local_path
        )


@patch("datasetinsights.io.gcs.GCSClient._is_file")
@patch("datasetinsights.io.gcs.GCSClient._download_file")
def test_gcs_client_download_file_url(mock_download_file, mock_is_file):
    url = base_url + file_name
    mocked_gcs_client = MagicMock()
    mock_is_file.return_value = True
    with patch(
        "datasetinsights.io.gcs.Client",
        MagicMock(return_value=mocked_gcs_client),
    ):
        client = GCSClient()
        client.download(local_path=local_path, url=url)
        mock_download_file.assert_called_with(
            mocked_gcs_client.get_bucket(), base_key + file_name, local_path
        )


@patch("datasetinsights.io.gcs.GCSClient._is_file")
@patch("datasetinsights.io.gcs.GCSClient._download_folder")
def test_gcs_client_download_folder_bucket_key(
    mock_download_folder, mock_is_file
):
    mocked_gcs_client = MagicMock()
    mock_is_file.return_value = False
    object_key = base_key
    with patch(
        "datasetinsights.io.gcs.Client",
        MagicMock(return_value=mocked_gcs_client),
    ):
        client = GCSClient()
        client.download(
            local_path=local_path, bucket=bucket_name, key=object_key
        )
        mock_download_folder.assert_called_with(
            mocked_gcs_client.get_bucket(), object_key, local_path
        )


@patch("datasetinsights.io.gcs.GCSClient._is_file")
@patch("datasetinsights.io.gcs.GCSClient._download_folder")
def test_gcs_client_download_folder_url(mock_download_folder, mock_is_file):
    mocked_gcs_client = MagicMock()
    mock_is_file.return_value = False
    with patch(
        "datasetinsights.io.gcs.Client",
        MagicMock(return_value=mocked_gcs_client),
    ):
        client = GCSClient()
        client.download(local_path=local_path, url=base_url)
        mock_download_folder.assert_called_with(
            mocked_gcs_client.get_bucket(), base_key, local_path
        )


@patch("datasetinsights.io.gcs.GCSClient._download_validate")
def test_download_folder(mock_download_validate):
    object_key = "path/to" + file_name
    mocked_gcs_client = MagicMock()
    with patch(
        "datasetinsights.io.gcs.Client",
        MagicMock(return_value=mocked_gcs_client),
    ):
        client = GCSClient()
        mocked_blob = MagicMock()
        mocked_bucket = MagicMock()
        mocked_bucket.list_blobs = MagicMock(return_value=[mocked_blob])
        mocked_blob.name = object_key
        client._download_folder(mocked_bucket, object_key, local_path)
        mock_download_validate.assert_called_with(mocked_blob, local_path)


@patch("datasetinsights.io.gcs.GCSClient._download_validate")
def test_download_file(mock_download_validate):
    object_key = base_key + file_name
    mocked_gcs_client = MagicMock()
    with patch(
        "datasetinsights.io.gcs.Client",
        MagicMock(return_value=mocked_gcs_client),
    ):
        client = GCSClient()
        mocked_bucket = MagicMock()
        mocked_blob = MagicMock()
        mocked_bucket.get_blob = MagicMock(return_value=mocked_blob)
        mocked_blob.name = object_key
        client._download_file(mocked_bucket, object_key, local_path)
        mocked_bucket.get_blob.assert_called_with(object_key)
        mock_download_validate.assert_called_with(
            mocked_blob, local_path + file_name
        )


@patch("datasetinsights.io.gcs.GCSClient._download_blob")
@patch("datasetinsights.io.gcs.GCSClient._checksum")
def test_download_validate(mock_checksum, mock_download_blob):
    mocked_gcs_client = MagicMock()
    with patch(
        "datasetinsights.io.gcs.Client",
        MagicMock(return_value=mocked_gcs_client),
    ):
        client = GCSClient()
        mocked_blob = MagicMock()
        client._download_validate(mocked_blob, local_path)
        mock_checksum.assert_called_with(mocked_blob, local_path)
        mock_download_blob.assert_called_with(mocked_blob, local_path)


@patch("datasetinsights.io.gcs.isdir")
def test_download_blob(mock_isdir):
    mocked_gcs_client = MagicMock()
    with patch(
        "datasetinsights.io.gcs.Client",
        MagicMock(return_value=mocked_gcs_client),
    ):
        mock_isdir.return_value = True
        object_key = base_key + file_name
        client = GCSClient()
        mocked_blob = MagicMock()
        mocked_blob.name = object_key
        mocked_download_blob = MagicMock()
        mocked_blob.download_to_filename = mocked_download_blob

        client._download_blob(mocked_blob, local_path)
        mocked_blob.download_to_filename.assert_called_with(local_path)


@patch("datasetinsights.io.gcs.GCSClient._md5_hex")
@patch("datasetinsights.io.gcs.validate_checksum")
def test_checksum(mock_checksum, mock_md5_hex):
    local_file_path = local_path + file_name
    mocked_gcs_client = MagicMock()
    mock_md5_hex.return_value = md5_hash_hex
    with patch(
        "datasetinsights.io.gcs.Client",
        MagicMock(return_value=mocked_gcs_client),
    ):
        client = GCSClient()
        mocked_bucket = MagicMock()
        mocked_blob = MagicMock()
        mocked_gcs_client.get_bucket = MagicMock(return_value=mocked_bucket)
        mocked_bucket.get_blob = MagicMock(return_value=mocked_blob)
        mocked_blob.md5_hash = md5_hash
        client._checksum(mocked_blob, local_file_path)
        mock_checksum.assert_called_with(
            local_file_path, md5_hash_hex, algorithm="MD5"
        )


@patch("datasetinsights.io.gcs.os.remove")
@patch("datasetinsights.io.gcs.validate_checksum")
def test_checksum_error(mock_checksum, mock_remove):
    local_file_path = local_path + file_name
    mocked_gcs_client = MagicMock()
    with patch(
        "datasetinsights.io.gcs.Client",
        MagicMock(return_value=mocked_gcs_client),
    ):
        client = GCSClient()
        mocked_bucket = MagicMock()
        mocked_blob = MagicMock()
        mocked_gcs_client.get_bucket = MagicMock(return_value=mocked_bucket)
        mocked_bucket.get_blob = MagicMock(return_value=mocked_blob)
        mocked_blob.md5_hash = md5_hash
        client._MD5_hex = MagicMock(return_value=md5_hash_hex)
        client._checksum(mocked_blob, local_file_path)

        mock_checksum.side_effect = ChecksumError
        with pytest.raises(ChecksumError):
            client._checksum(mocked_blob, local_file_path)
            mock_remove.assert_called_once()


def test_is_file():
    object_key = base_key + file_name
    mocked_gcs_client = MagicMock()
    with patch(
        "datasetinsights.io.gcs.Client",
        MagicMock(return_value=mocked_gcs_client),
    ):
        client = GCSClient()
        mocked_bucket = MagicMock()
        mocked_blob = MagicMock()
        mocked_bucket.get_blob = MagicMock(return_value=mocked_blob)
        mocked_blob.name = object_key
        actual_result = client._is_file(mocked_bucket, object_key)
        mocked_bucket.get_blob.assert_called_with(object_key)
        expected_result = True
        assert actual_result == expected_result


def test_MD5_hex():
    mocked_gcs_client = MagicMock()
    with patch(
        "datasetinsights.io.gcs.Client",
        MagicMock(return_value=mocked_gcs_client),
    ):
        client = GCSClient()
        actual_result = client._md5_hex(md5_hash)
        expected_result = "69b7"
        assert actual_result == expected_result


def test_upload_file():
    localfile = local_path + file_name
    mocked_gcs_client = MagicMock()
    with patch(
        "datasetinsights.io.gcs.Client",
        MagicMock(return_value=mocked_gcs_client),
    ):
        client = GCSClient()
        mocked_bucket = MagicMock()
        mocked_blob = MagicMock()
        mocked_gcs_client.get_bucket = MagicMock(return_value=mocked_bucket)
        mocked_bucket.blob = MagicMock(return_value=mocked_blob)
        mocked_blob.upload_from_filename = MagicMock()

        client._upload_file(
            local_path=localfile, bucket=mocked_bucket, key=base_key
        )
        mocked_blob.upload_from_filename.assert_called_with(localfile)


@patch("datasetinsights.io.gcs.Path.glob")
def test_upload_folder(mock_glob):
    localfile = local_path + file_name
    mocked_gcs_client = MagicMock()
    mock_glob.return_value = [Path(localfile)]
    with patch(
        "datasetinsights.io.gcs.Client",
        MagicMock(return_value=mocked_gcs_client),
    ):
        client = GCSClient()
        client._upload_file = MagicMock()
        mocked_bucket = MagicMock()
        mocked_blob = MagicMock()
        mocked_gcs_client.get_bucket = MagicMock(return_value=mocked_bucket)
        mocked_bucket.blob = MagicMock(return_value=mocked_blob)
        mocked_blob.upload_from_filename = MagicMock()
        client._upload_folder(
            local_path=local_path, bucket=mocked_bucket, key=base_key
        )
        client._upload_file.assert_called_with(
            bucket=mocked_bucket,
            key=base_key + file_name,
            local_path=localfile,
        )


def test_gcs_downloader():
    url = "gs://fake_bucket/path/to"
    mocked_gcs_client = MagicMock()
    with patch(
        "datasetinsights.io.downloader.gcs_downloader.GCSClient",
        MagicMock(return_value=mocked_gcs_client),
    ):

        downloader = GCSDatasetDownloader()
        downloader.download(url, local_path)
        mocked_gcs_client.download.assert_called_with(
            local_path=local_path, url=url
        )


def test_gcs_parse():
    mocked_gcs_client = MagicMock()
    with patch(
        "datasetinsights.io.gcs.Client",
        MagicMock(return_value=mocked_gcs_client),
    ):
        client = GCSClient()
        th_bucket = "some_bucket_name"
        th_path = "some/cloud/path"
        url = "gs://some_bucket_name/some/cloud/path"

        bucket, path = client._parse(url)
        assert (bucket, path) == (th_bucket, th_path)

        bad_url = "s3://path/to/bad/url"
        with pytest.raises(ValueError, match=r"Specified destination prefix:"):
            client._parse(bad_url)
