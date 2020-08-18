import logging
import os
import tempfile
import zipfile

from datasetinsights.io.download import download_file, validate_checksum
from datasetinsights.io.downloader.base import DatasetDownloader
from datasetinsights.io.exceptions import ChecksumError

logger = logging.getLogger(__name__)

DATASETS = ["synthetic", "groceries_real"]


class HTTPDownloader(DatasetDownloader, protocol="http://"):
    def download(self, source_uri, output, checksum_file=None, **kwargs):

        dataset_path = os.path.join(output, "dataset.zip")
        download_file(source_uri, dataset_path)

        if checksum_file:

            checksum = HTTPDownloader.get_checksum_from_file(checksum_file)

            try:
                validate_checksum(dataset_path, checksum)
            except ChecksumError as e:
                logger.info("Checksum mismatch. Delete the downloaded file.")
                os.remove(dataset_path)
                raise e

        HTTPDownloader.unzip_file(dataset_path, output)

    @staticmethod
    def get_checksum_from_file(filepath):

        if filepath.startswith(("http://", "https://")):
            with tempfile.TemporaryDirectory() as tmp:
                checksum_file_path = os.path.join(tmp, "checksum.txt")
                download_file(source_uri=filepath, dest_path=checksum_file_path)
                return HTTPDownloader.read_checksum_from_txt(checksum_file_path)

        elif filepath.startswith("/"):
            checksum_file_path = filepath
            return HTTPDownloader.read_checksum_from_txt(checksum_file_path)

        else:
            raise ValueError(f"Can not get checksum from path: {filepath}")

    @staticmethod
    def read_checksum_from_txt(filepath):
        with open(filepath, "rb") as file:
            checksum = file.read()
        return int(checksum)

    @staticmethod
    def unzip_file(filepath, destination):
        """Unzips a zip file to the destination and delete the zip file.

        Args:
            filepath (str): File path of the zip file.
            destination (str): Path where to unzip contents of zipped file.
        """
        with zipfile.ZipFile(filepath) as file:
            logger.info(f"Unzipping file: {filepath} to {destination}")
            file.extractall(destination)
