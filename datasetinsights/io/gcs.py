import base64
import logging
import os
from os import makedirs
from os.path import basename, isdir
from pathlib import Path

from google.cloud.storage import Client

from datasetinsights.io.download import validate_checksum
from datasetinsights.io.exceptions import ChecksumError, DownloadError

logger = logging.getLogger(__name__)


class GCSClient:
    """ This class is used to download data from GCS location
        and perform function such as downloading the dataset and checksum
        validation.
    """

    def __init__(self, **kwargs):
        """ Initialize a client to google cloud storage (GCS).
        """
        self.client = Client(**kwargs)

    def download(self, *, url=None, local_path=None, bucket=None, key=None):
        """ This method is used to download the dataset from GCS.

        Args:
            url (str): This is the downloader-uri that indicates where
                              the dataset should be downloaded from.

            local_path (str): This is the path to the directory where the
                          download will store the dataset.

            bucket (str): gcs bucket name
            key (str): object key path

            Examples:
                >>> url = "gs://bucket/folder or gs://bucket/folder/data.zip"
                >>> local_path = "/tmp/folder"
                >>> bucket ="bucket"
                >>> key ="folder/data.zip" or "folder"

        """
        if not (bucket and key) and url:
            bucket, key = parse_gcs_location(url)

        bucket_obj = self.client.get_bucket(bucket)
        if self._is_file(bucket_obj, key):
            self._download_file(bucket_obj, key, local_path)
        else:
            self._download_folder(bucket_obj, key, local_path)

    def upload(self, localfile, bucket_name, object_key):
        """ Upload a single object to GCS
        """
        bucket = self.client.get_bucket(bucket_name)
        blob = bucket.blob(object_key)

        blob.upload_from_filename(localfile)

    def _download_folder(self, bucket, key, local_path):
        """ download all files from directory
        """
        blobs = bucket.list_blobs(prefix=key)
        for blob in blobs:
            local_file_path = blob.name.replace(key, local_path)
            self._download_validate(blob, key, local_file_path)

    def _download_file(self, bucket, key, local_path):
        """ download single file
        """
        blob = bucket.get_blob(key)
        key_suffix = key.replace("/" + basename(key), "")
        local_file_path = blob.name.replace(key_suffix, local_path)
        self._download_validate(blob, key, local_file_path)

    def _download_validate(self, blob, key, local_file_path):
        """ download file and validate checksum
        """
        dst_dir = local_file_path.replace("/" + basename(local_file_path), "")
        self._download_blob(blob, dst_dir, key, local_file_path)
        self._checksum(blob, local_file_path)

    def _download_blob(self, blob, dst_dir, key, local_file_path):
        """ download blob from gcs
        Raises:
            ChecksumError: This will raise this error if checksum doesn't
                           matches
        """
        if not isdir(dst_dir):
            makedirs(dst_dir)
        try:
            logger.info(f"Downloading from {key} to {local_file_path}.")
            blob.download_to_filename(local_file_path)
        except DownloadError as e:
            logger.info(
                f"The request download from {key} -> {local_file_path} can't "
                f"be completed."
            )
            raise e

    def _checksum(self, blob, dst_file_name):
        """validate checksum"""
        expected_checksum = blob.md5_hash
        if expected_checksum:
            expected_checksum_hex = self._MD5_hex(expected_checksum)
            try:
                validate_checksum(
                    dst_file_name, expected_checksum_hex, algorithm="MD5"
                )
            except ChecksumError as e:
                logger.info("Checksum mismatch. Delete the downloaded files.")
                os.remove(dst_file_name)
                raise e

    def _is_file(self, bucket, key):
        """given key is file or directory"""
        blob = bucket.get_blob(key)
        if blob:
            return blob.name == key
        return False

    def _MD5_hex(self, checksum):
        """fix the missing padding if requires and converts into hex"""
        missing_padding = len(checksum) % 4
        if missing_padding != 0:
            checksum += "=" * (4 - missing_padding)
        return base64.b64decode(checksum).hex()


def parse_gcs_location(url):
    """Split an GCS-prefixed URL into bucket and path."""
    gcs_prefix = "gs://"
    key_separator = "/"
    if not url.startswith(gcs_prefix):
        raise ValueError(
            f"Specified destination prefix: {url} does not start "
            f"with {gcs_prefix}"
        )
    url = url[len(gcs_prefix) :]
    if key_separator not in url:
        raise ValueError(
            f"Specified destination prefix: {gcs_prefix + url} does "
            f"not have object key "
        )
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
    bucket, prefix = parse_gcs_location(cloud_path)
    for path in Path(folder).glob(pattern):
        if path.is_dir():
            continue
        full_path = str(path)
        relative_path = str(path.relative_to(folder))
        object_key = os.path.join(prefix, relative_path)
        client.upload(full_path, bucket, object_key)
