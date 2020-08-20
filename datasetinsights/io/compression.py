import gzip
import logging
import os
import tarfile
import zipfile

import filetype

logger = logging.getLogger(__name__)


class ZipFileCompression:
    @staticmethod
    def decompress(filepath, destination):
        """ Unzips a zip file to the destination.

        Args:
            filepath (str): File path of the zip file.
            destination (str): Path where to unzip contents of zipped file.

        """
        with zipfile.ZipFile(filepath, "r") as file:
            logger.info(f"Unzipping file: {filepath} to {destination}")
            file.extractall(destination)


class TarFileCompression:
    @staticmethod
    def decompress(filepath, destination):
        """Decompress a tar file to the destination.

        Args:
            filepath (str): File path of the tar file.
            destination (str): Path where to decompress contents of tar file.

        """
        with tarfile.open(filepath, "r") as tar:
            logger.info(f"Decompressing file: {filepath} to {destination}")
            tar.extractall(destination)


class GZipCompression:
    @staticmethod
    def decompress(filepath, destination):
        """ Decompress a gzip file to the destination.

        This will decompress a gzip file of name 'file.txt.gz' to 'file.txt'

        Args:
            filepath (str): File path of the gzip file.
            destination (str): Path where to decompress contents of gzip file.

        """
        destination = os.path.join(
            destination, os.path.splitext(os.path.basename(filepath))[0]
        )
        with open(destination, "wb") as out_f, gzip.GzipFile(filepath) as zip_f:
            out_f.write(zip_f.read())


def get_file_type_from_filepath(filepath):
    """ Detects file type of a file.

    Args:
        filepath (str): File path of the file.

    Returns: a filetype <https://pypi.org/project/filetype/> object.

    """
    file_type = filetype.guess(filepath)
    if file_type is None:
        raise ValueError(f"Can not detect file type from path: {filepath}")
    return file_type


def compression_factory(filepath):
    """ Get compression class from filepath.

    Args:
        filepath (str): File path of the file.

    Returns: Compression class object.
    """
    file_type = get_file_type_from_filepath(filepath)
    compression_class = {
        "zip": ZipFileCompression,
        "gz": GZipCompression,
        "tar": TarFileCompression,
    }
    try:
        compressor = compression_class[file_type.extension]
    except KeyError:
        raise ValueError(
            f"Compression type: {file_type.mime} not supported " f"currently."
        )

    return compressor
