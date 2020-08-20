import logging
import os
from os import makedirs
from os.path import basename, isdir
from pathlib import Path

from google.cloud.storage import Client

from datasetinsights.io.download import validate_checksum
from datasetinsights.io.exceptions import ChecksumError, DownloadError

logger = logging.getLogger(__name__)

MD5 = "MD5"
REPAD = "=="


class GCSClient:
    def __init__(self, **kwargs):
        """ Initialize a client to google cloud storage (GCS).
        """
        self.client = Client(**kwargs)

    def download(self, bucket_name, object_key, localfile):
        """ Download a single object from GCS
        """
        bucket = self.client.get_bucket(bucket_name)
        blob = bucket.blob(object_key)

        blob.download_to_filename(localfile)

    def upload(self, localfile, bucket_name, object_key):
        """ Upload a single object to GCS
        """
        bucket = self.client.get_bucket(bucket_name)
        blob = bucket.blob(object_key)

        blob.upload_from_filename(localfile)


def gcs_bucket_and_path(url):
    """Split an GCS-prefixed URL into bucket and path."""
    gcs_prefix = "gs://"
    if not url.startswith(gcs_prefix):
        raise ValueError(
            f"Specified destination prefix: {url} does not start "
            f"with {gcs_prefix}"
        )
    url = url[len(gcs_prefix) :]
    idx = url.index("/")
    bucket = url[:idx]
    path = url[(idx + 1) :]

    return bucket, path


def copy_folder_to_gcs(cloud_path, folder, pattern="*"):
    """Copy all files within a folder to GCS

    Args:
        pattern: Unix glob patterns. Use **/* for recursive glob.
    """
    client = GCSClient()
    bucket, prefix = gcs_bucket_and_path(cloud_path)
    for path in Path(folder).glob(pattern):
        if path.is_dir():
            continue
        full_path = str(path)
        relative_path = str(path.relative_to(folder))
        object_key = os.path.join(prefix, relative_path)
        client.upload(full_path, bucket, object_key)


def download_file_from_gcs(cloud_path, local_path, filename, use_cache=True):
    """Helper method to download a single file from GCS

    Args:
        cloud_path: Full path to a GCS folder
        local_path: Local path to a folder where the file should be stored
        filename: The filename to be downloaded

    Returns:
        str: Full path to the downloaded file

    Examples:
        >>> cloud_path = "gs://bucket/folder"
        >>> local_path = "/tmp/folder"
        >>> filename = "file.txt"
        >>> download_file_from_gcs(cloud_path, local_path, filename)
        # download file gs://bucket/folder/file.txt to /tmp/folder/file.txt
    """
    bucket, prefix = gcs_bucket_and_path(cloud_path)
    object_key = os.path.join(prefix, filename)
    local_filepath = os.path.join(local_path, filename)

    path = Path(local_path)
    path.mkdir(parents=True, exist_ok=True)
    client = GCSClient()

    if os.path.exists(local_filepath) and use_cache:
        logger.info(
            f"Found existing file in {local_filepath}. Skipping download."
        )
    else:
        logger.info(
            f"Downloading from {cloud_path}/{filename} to {local_filepath}."
        )
        client.download(bucket, object_key, local_filepath)

    # TODO(YC) Should run file checksum before return.
    return local_filepath


def download_folder_from_gcs(cloud_path, local_path):
    """Helper method to download list of files from GCS

    Args:
        cloud_path: Full path to a GCS folder
        local_path: Local path to a folder where the file should be stored

    Returns:
        str: Full path to the downloaded file

    Examples:
        >>> cloud_path = "gs://bucket/folder"
        >>> local_path = "/tmp/folder"
        >>> filename = "file.txt"
        >>> download_file_from_gcs(cloud_path, local_path, filename)
        # download file gs://bucket/folder/file.txt to /tmp/folder/file.txt
    """
    bucket, prefix = gcs_bucket_and_path(cloud_path)
    client = GCSClient().client
    bucket = client.get_bucket(bucket)
    blobs = bucket.list_blobs(prefix=prefix)
    for blob in blobs:
        blob_name = blob.name
        dst_file_name = blob_name.replace(prefix, local_path)
        dst_dir = dst_file_name.replace("/" + basename(dst_file_name), "")
        if not isdir(dst_dir):
            makedirs(dst_dir)
        try:
            logger.info(f"Downloading from {prefix} to {dst_file_name}.")
            blob.download_to_filename(dst_file_name)
        except DownloadError as e:
            logger.info(
                f"The request download from {prefix} -> {dst_file_name} can't "
                f"be completed."
            )
            raise e
        expected_checksum = blob.md5_hash
        if expected_checksum:
            expected_checksum += REPAD
            try:
                validate_checksum(
                    dst_file_name, expected_checksum, algorithm=MD5
                )
            except ChecksumError as e:
                logger.info("Checksum mismatch. Delete the downloaded files.")
                os.remove(dst_file_name)
                raise e
