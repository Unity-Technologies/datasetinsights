from unittest.mock import MagicMock, patch

import pytest

from datasetinsights.io.gcs import GCSClient, gcs_bucket_and_path


def test_gcs_client_warpper():
    bucket_name = "fake_bucket"
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

        mocked_blob.download_to_filename = MagicMock()
        client.download(bucket_name, object_key, localfile)
        mocked_gcs_client.get_bucket.assert_called_with(bucket_name)
        mocked_bucket.blob.assert_called_with(object_key)
        mocked_blob.download_to_filename.assert_called_with(localfile)

        mocked_blob.upload_from_filename = MagicMock()
        client.upload(localfile, bucket_name, object_key)
        mocked_blob.upload_from_filename.assert_called_with(localfile)


def test_gcs_bucket_and_path():
    th_bucket = "some_bucket_name"
    th_path = "some/cloud/path"
    url = "gs://some_bucket_name/some/cloud/path"

    bucket, path = gcs_bucket_and_path(url)
    assert (bucket, path) == (th_bucket, th_path)

    bad_url = "s3://path/to/bad/url"
    with pytest.raises(ValueError, match=r"Specified destination prefix:"):
        gcs_bucket_and_path(bad_url)
