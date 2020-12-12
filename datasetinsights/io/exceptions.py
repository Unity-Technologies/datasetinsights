class DownloadError(Exception):
    """ Raise when download file failed.
    """


class ChecksumError(Exception):
    """ Raises when the downloaded file checksum is not correct.
    """


class InvalidTrackerError(Exception):
    """ Raises when unknown tracker requested .
    """


class InvalidOverrideKey(Exception):
    """ Raises when non existence key requested to override .
    """
