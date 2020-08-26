class DownloadError(Exception):
    """ Raise when download file failed.
    """


class ChecksumError(Exception):
    """ Raises when the downloaded file checksum is not correct.
    """


class UploadError(Exception):
    """ Raise when upload file failed.
    """
