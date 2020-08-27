from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from datasetinsights.io.downloader import GCSDatasetDownloader
from datasetinsights.io.exceptions import (
    ChecksumError,
    DownloadError,
    UploadError,
)
from datasetinsights.io.gcs import GCSClient

bucket_name = "fake_bucket"
local_path = "path/to/local"
md5_hash = "abc=="
md5_hash_hex = "12345"
file_name = "/data.zip"


@patch("datasetinsights.io.gcs.isdir")
def test_gcs_client_upload_file(mock_isdir):
    object_key = "path/to/object"
    localfile = "path/to/local/file" + file_name
    mocked_gcs_client = MagicMock()
    mock_isdir.return_value = False
    url = "gs://fake_bucket/path/to" + file_name
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
        client.upload(local_path=localfile, bucket=bucket_name, key=object_key)
        mocked_blob.upload_from_filename.assert_called_with(localfile)

        client.upload(local_path=localfile, url=url)
        mocked_gcs_client.get_bucket.assert_called_with(bucket_name)
        assert mocked_blob.upload_from_filename.call_count == 2


@patch("datasetinsights.io.gcs.Path.glob")
@patch("datasetinsights.io.gcs.isdir")
def test_gcs_client_upload_folder(mock_isdir, mock_glob):
    object_key = "path/to/object"
    localfile = local_path + file_name
    mocked_gcs_client = MagicMock()
    mock_isdir.return_value = True
    mock_glob.return_value = [Path(localfile)]
    url = "gs://fake_bucket/path/to"
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
        client.upload(local_path=local_path, bucket=bucket_name, key=object_key)
        client._upload_file.assert_called_with(
            bucket=mocked_bucket,
            key=object_key + file_name,
            local_path=localfile,
        )

        client.upload(local_path=local_path, url=url)
        mocked_gcs_client.get_bucket.assert_called_with(bucket_name)
        assert mocked_blob.upload_from_filename.call_count == 2


@patch("datasetinsights.io.gcs.isdir")
@patch("datasetinsights.io.gcs.validate_checksum")
def test_gcs_client_download_file(mock_checksum, mock_isdir):
    object_key = "path/to" + file_name
    local_file_path = local_path + file_name
    url = "gs://fake_bucket/path/to" + file_name
    mocked_gcs_client = MagicMock()
    mock_isdir.return_value = True
    with patch(
        "datasetinsights.io.gcs.Client",
        MagicMock(return_value=mocked_gcs_client),
    ):
        client = GCSClient()
        mocked_bucket = MagicMock()
        mocked_blob = MagicMock()
        mocked_gcs_client.get_bucket = MagicMock(return_value=mocked_bucket)
        mocked_bucket.get_blob = MagicMock(return_value=mocked_blob)
        mocked_blob.name = object_key
        mocked_blob.md5_hash = md5_hash
        client._is_file = MagicMock(return_value=True)
        client._MD5_hex = MagicMock(return_value=md5_hash_hex)
        client.download(
            local_path=local_path, bucket=bucket_name, key=object_key
        )
        mocked_gcs_client.get_bucket.assert_called_with(bucket_name)
        mocked_blob.download_to_filename.assert_called_with(local_file_path)

        mock_checksum.assert_called_once()
        mock_checksum.assert_called_with(
            local_file_path, md5_hash_hex, algorithm="MD5"
        )

        client.download(local_path=local_path, url=url)
        mocked_gcs_client.get_bucket.assert_called_with(bucket_name)
        mocked_blob.download_to_filename.assert_called_with(local_file_path)

        mock_checksum.assert_called_with(
            local_file_path, md5_hash_hex, algorithm="MD5"
        )
        assert mock_checksum.call_count == 2


@patch("datasetinsights.io.gcs.isdir")
@patch("datasetinsights.io.gcs.validate_checksum")
def test_gcs_client_download_folder(mock_checksum, mock_isdir):
    object_key = "path/to"
    local_file_path = local_path + file_name
    url = "gs://fake_bucket/path/to"
    mock_isdir.return_value = True
    mocked_gcs_client = MagicMock()
    with patch(
        "datasetinsights.io.gcs.Client",
        MagicMock(return_value=mocked_gcs_client),
    ):
        client = GCSClient()
        client._is_file = MagicMock(return_value=False)
        client._MD5_hex = MagicMock(return_value=md5_hash_hex)
        mocked_bucket = MagicMock()
        mocked_blob = MagicMock()
        mocked_gcs_client.get_bucket = MagicMock(return_value=mocked_bucket)
        mocked_bucket.list_blobs = MagicMock(return_value=[mocked_blob])
        mocked_blob.name = object_key + file_name
        mocked_blob.md5_hash = md5_hash
        client.download(
            local_path=local_path, bucket=bucket_name, key=object_key
        )
        mocked_gcs_client.get_bucket.assert_called_with(bucket_name)
        mocked_blob.download_to_filename.assert_called_with(local_file_path)

        mock_checksum.assert_called_once()
        mock_checksum.assert_called_with(
            local_file_path, md5_hash_hex, algorithm="MD5"
        )

        client.download(local_path=local_path, url=url)
        mocked_gcs_client.get_bucket.assert_called_with(bucket_name)
        mocked_blob.download_to_filename.assert_called_with(local_file_path)

        mock_checksum.assert_called_with(
            local_file_path, md5_hash_hex, algorithm="MD5"
        )
        assert mock_checksum.call_count == 2


def test_download_folder():
    object_key = "path/to" + file_name
    mocked_gcs_client = MagicMock()
    with patch(
        "datasetinsights.io.gcs.Client",
        MagicMock(return_value=mocked_gcs_client),
    ):
        client = GCSClient()
        mocked_bucket = MagicMock()
        mocked_blob = MagicMock()
        mocked_bucket.list_blobs = MagicMock(return_value=[mocked_blob])
        mocked_blob.name = object_key
        mocked_download_validate = MagicMock()
        client._download_validate = mocked_download_validate
        client._download_folder(mocked_bucket, object_key, local_path)
        mocked_bucket.list_blobs.assert_called_with(prefix=object_key)
        mocked_download_validate.assert_called_with(mocked_blob, local_path)


def test_download_file():
    object_key = "path/to" + file_name
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
        mocked_download_validate = MagicMock()
        client._download_validate = mocked_download_validate
        client._download_file(mocked_bucket, object_key, local_path)
        mocked_bucket.get_blob.assert_called_with(object_key)
        mocked_download_validate.assert_called_with(
            mocked_blob, local_path + file_name
        )


def test_download_validate():
    mocked_gcs_client = MagicMock()
    with patch(
        "datasetinsights.io.gcs.Client",
        MagicMock(return_value=mocked_gcs_client),
    ):
        client = GCSClient()
        mocked_blob = MagicMock()
        mocked_download_blob = MagicMock()
        mocked_checksum = MagicMock()
        client._download_blob = mocked_download_blob
        client._checksum = mocked_checksum

        client._download_validate(mocked_blob, local_path)
        mocked_checksum.assert_called_with(mocked_blob, local_path)
        mocked_download_blob.assert_called_with(mocked_blob, local_path)


@patch("datasetinsights.io.gcs.isdir")
def test_download_blob(mock_isdir):
    mocked_gcs_client = MagicMock()
    with patch(
        "datasetinsights.io.gcs.Client",
        MagicMock(return_value=mocked_gcs_client),
    ):
        mock_isdir.return_value = True
        object_key = "path/to" + file_name
        client = GCSClient()
        mocked_blob = MagicMock()
        mocked_blob.name = object_key
        mocked_download_blob = MagicMock()
        mocked_blob.download_to_filename = mocked_download_blob

        client._download_blob(mocked_blob, local_path)
        mocked_blob.download_to_filename.assert_called_with(local_path)

        mocked_blob.download_to_filename.side_effect = DownloadError
        with pytest.raises(DownloadError):
            client._download_blob(mocked_blob, local_path)
            assert mocked_download_blob.call_count == 2


@patch("datasetinsights.io.gcs.os.remove")
@patch("datasetinsights.io.gcs.validate_checksum")
def test_checksum(mock_checksum, mock_remove):
    object_key = "path/to" + file_name
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
        mocked_blob.name = object_key
        mocked_blob.md5_hash = md5_hash
        client._MD5_hex = MagicMock(return_value=md5_hash_hex)
        client._checksum(mocked_blob, local_file_path)
        mock_checksum.assert_called_once()
        mock_checksum.assert_called_with(
            local_file_path, md5_hash_hex, algorithm="MD5"
        )

        mock_checksum.side_effect = ChecksumError
        with pytest.raises(ChecksumError):
            client._checksum(mocked_blob, local_file_path)
            mock_remove.assert_called_once()


def test_is_file():
    object_key = "path/to" + file_name
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
        actual_result = client._MD5_hex(md5_hash)
        expected_result = "69b7"
        assert actual_result == expected_result


def test_upload_file():
    object_key = "path/to/object"
    localfile = "path/to/local/file"
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
            local_path=localfile, bucket=mocked_bucket, key=object_key
        )
        mocked_blob.upload_from_filename.assert_called_with(localfile)

        mocked_blob.upload_from_filename.side_effect = UploadError
        with pytest.raises(UploadError):
            client._upload_file(
                local_path=localfile, bucket=mocked_bucket, key=object_key
            )
            assert mocked_blob.upload_from_filename.call_count == 2


@patch("datasetinsights.io.gcs.Path.glob")
def test_upload_folder(mock_glob):
    object_key = "path/to/object"
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
            local_path=local_path, bucket=mocked_bucket, key=object_key
        )
        client._upload_file.assert_called_with(
            bucket=mocked_bucket,
            key=object_key + file_name,
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
