import logging
import shutil

import filetype

logger = logging.getLogger(__name__)


def decompress(filepath, destination):
    try:
        extension = _get_file_extension_from_filepath(filepath)
        shutil.unpack_archive(filepath, destination, extension)
    except ValueError as e:
        logger.debug(f"Current file format is not supported for decompression.")
        raise e


def _get_file_extension_from_filepath(filepath):
    """ Detects file type of a file.
    See  <https://pypi.org/project/filetype/> for more info.

    Args:
        filepath (str): File path of the file.

    Returns: Extension of the file.

    """
    file_type_obj = filetype.guess(filepath)
    if file_type_obj is None:
        raise ValueError(f"Can not detect file type from path: {filepath}")
    return file_type_obj.extension
