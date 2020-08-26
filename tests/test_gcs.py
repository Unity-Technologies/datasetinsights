from pathlib import Path
from unittest.mock import MagicMock, patch

from datasetinsights.io.downloader import GCSDatasetDownloader
from datasetinsights.io.gcs import GCSClient

bucket_name = "fake_bucket"
local_path = "data/io"
md5_hash = "abc=="
md5_hash_hex = "12345"


@patch("datasetinsights.io.gcs.os.path.isdir")
def test_gcs_client_upload_file(mock_isdir):
    object_key = "path/to/object"
    localfile = "path/to/local/file"
    mocked_gcs_client = MagicMock()
    mock_isdir.return_value = False
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


@patch("datasetinsights.io.gcs.Path.glob")
@patch("datasetinsights.io.gcs.os.path.isdir")
def test_gcs_client_upload_folder(mock_isdir, mock_glob):
    object_key = "path/to/object"
    localfile = "data/io/data.zip"
    mocked_gcs_client = MagicMock()
    mock_isdir.return_value = True
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
        client.upload(local_path=local_path, bucket=bucket_name, key=object_key)
        client._upload_file.assert_called_with(
            bucket=mocked_bucket,
            key=object_key + "/data.zip",
            local_path=localfile,
        )


@patch("datasetinsights.io.gcs.validate_checksum")
def test_gcs_client_download_file(mock_checksum):
    object_key = "path/to/object.zip"
    local_file_path = "data/io/object.zip"
    url = "gs://fake_bucket/path/to/object.zip"
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


@patch("datasetinsights.io.gcs.validate_checksum")
def test_gcs_client_download_folder(mock_checksum):
    object_key = "path/to"
    local_file_path = "data/io/object.zip"
    url = "gs://fake_bucket/path/to"

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
        mocked_blob.name = object_key + "/object.zip"
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
