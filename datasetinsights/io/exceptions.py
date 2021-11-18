class DownloadError(Exception):
    """ Raise when download file failed.
    """


class ChecksumError(Exception):
    """ Raises when the downloaded file checksum is not correct.
    """


class InvalidTrackerError(Exception):
    """ Raises when unknown tracker requested .
    """


class InvalidCOCOImageIdError(Exception):
    """ Raised when invalid image id is given.
    """


class InvalidCOCOCategoryIdError(Exception):
    """ Raised when invalid category id is given.
    """
