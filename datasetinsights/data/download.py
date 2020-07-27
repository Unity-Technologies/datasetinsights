import logging
import os
import zlib
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from .exceptions import DownloadError

logger = logging.getLogger(__name__)

# Timeout of requests (in seconds)
DEFAULT_TIMEOUT = 1800
# Retry after failed request
DEFAULT_MAX_RETRIES = 5


class TimeoutHTTPAdapter(HTTPAdapter):
    def __init__(self, timeout, *args, **kwargs):
        self.timeout = timeout
        super().__init__(*args, **kwargs)

    def send(self, request, **kwargs):
        kwargs["timeout"] = self.timeout
        return super().send(request, **kwargs)


def download_file(source_uri: str, dest_path: str, use_cache: bool = True):
    """Download a file specified from a source uri

    Args:
        source_uri (str): source url where the file should be downloaded
        dest_path (str): destination path of the file
        use_cache (bool): use_cache (bool): use cache instead of
                re-download if file exists

    Returns:
        String of destination path.
    """
    dest_path = Path(dest_path)
    if dest_path.exists() and use_cache:
        return dest_path

    logger.debug(f"Trying to download file from {source_uri} -> {dest_path}")
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
                f"The request download from {source_uri} -> {dest_path} can't "
                f"be completed."
            )

            raise DownloadError(err_msg)
        else:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_path, "wb") as f:
                f.write(response.content)

    return dest_path


def compare_checksums(file_path, checksum_path):
    """Compare checksums for source and destination file.
    Will raise error if two checksums are different.

    Args:
        file_path (str): local path of the file
        checksum_path (str): checksum file for the source file

    Returns:
        bool: whether it passes or fails
    """
    source_file_checksum = _get_source_checksum(checksum_path)
    local_file_checksum = _get_local_checksum(file_path)
    if local_file_checksum != source_file_checksum:
        os.remove(checksum_path)
        os.remove(file_path)
        return False
    return True


def _get_local_checksum(local_path):
    """Calculate checksum (CRC32) for a local file

    Args:
        local_path (str): local path of the file

    Returns:
        str: checksum for the local file
    """
    with open(local_path, "rb") as f:
        local_file_crc32 = zlib.crc32(f.read())

    return str(local_file_crc32)


def _get_source_checksum(checksum_path):
    """Get the checksum for the source file

    Args:
        checksum_path (str): downloaded checksum file path

    Returns:
        str: checksum for the source file
    """
    with open(checksum_path, "r") as f:
        source_checksum = f.read()

    return source_checksum
