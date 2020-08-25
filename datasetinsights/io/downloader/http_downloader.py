import logging
import os
import re
from pathlib import Path

import requests
from requests.packages.urllib3.util.retry import Retry

from datasetinsights.io.download import (
    TimeoutHTTPAdapter,
    get_checksum_from_file,
    validate_checksum,
)
from datasetinsights.io.downloader.base import DatasetDownloader
from datasetinsights.io.exceptions import ChecksumError, DownloadError

# number of workers for ThreadPoolExecutor. This is the default value
# in python3.8
MAX_WORKER = min(32, os.cpu_count() + 4)
# Timeout of requests (in seconds)
DEFAULT_TIMEOUT = 1800
# Retry after failed request
DEFAULT_MAX_RETRIES = 5

logger = logging.getLogger(__name__)


class HTTPDatasetDownloader(DatasetDownloader, protocol="http://"):
    """ This class is used to download data from any HTTP or HTTPS public url
        and perform function such as downloading the dataset and checksum
        validation if checksum file path is provided.
    """

    def download(self, source_uri, output, checksum_file=None, **kwargs):
        """ This method is used to download the dataset from HTTP or HTTPS url.

        Args:
            source_uri (str): This is the downloader-uri that indicates where
                              the dataset should be downloaded from.

            output (str): This is the path to the directory where the download
                          will store the dataset.

            checksum_file (str): This is path of the txt file that contains
                                 checksum of the dataset to be downloaded. It
                                 can be HTTP or HTTPS url or local path.

        Raises:
            ChecksumError: This will raise this error if checksum doesn't
                           matches

        """
        dataset_path = download_dataset_from_http_url(source_uri, output)

        if checksum_file:
            logger.debug("Reading checksum from checksum file.")
            checksum = get_checksum_from_file(checksum_file)
            try:
                logger.debug("Validating checksum!!")
                validate_checksum(dataset_path, int(checksum))
            except ChecksumError as e:
                logger.info("Checksum mismatch. Deleting the downloaded file.")
                os.remove(dataset_path)
                raise e


def download_dataset_from_http_url(source_uri, output):
    """ Downloads dataset from HTTP(S) URL and detects name of the file to be
        downloaded.

        Args
            source_uri (str): This is the downloader-uri that indicates where
                              the dataset should be downloaded from.

            output (str): This is the path to the directory where the
                          download will store the dataset.
    """
    output = Path(output)

    adapter = TimeoutHTTPAdapter(
        timeout=DEFAULT_TIMEOUT, max_retries=Retry(total=DEFAULT_MAX_RETRIES)
    )
    with requests.Session() as http:
        http.mount("https://", adapter)
        try:
            response = http.get(source_uri)
            response.raise_for_status()
        except requests.exceptions.RequestException as ex:
            logger.error(ex)
            err_msg = (
                f"The request download from {source_uri} -> {output} can't "
                f"be completed."
            )
            raise DownloadError(err_msg)
        else:
            output.parent.mkdir(parents=True, exist_ok=True)

            file_name = get_filename_from_response(response)
            if file_name is None:
                file_name = get_file_name_from_uri(source_uri)

            dataset_path = output / file_name

            with open(dataset_path, "wb") as f:
                f.write(response.content)
    return dataset_path


def get_filename_from_response(response):
    """ Gets filename from requests response object

        Args:
            response: requests.Response() object that contains the server's
            response to the HTTP request.

        Returns:
            filename (str): Name of the file to be downloaded
    """
    cd = response.headers.get("content-disposition")
    if not cd:
        return None
    file_name = re.findall("filename=(.+)", cd)
    if len(file_name) == 0:
        return None
    return file_name[0]


def get_file_name_from_uri(uri):
    """ Gets filename from URI

    Args:
        uri (str): URI

    """
    return uri.split("/")[-1]
