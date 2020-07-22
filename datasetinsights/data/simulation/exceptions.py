class DownloadError(Exception):
    """ Raise when download file failed.
    """


class DefinitionIDError(Exception):
    """ Raise when a given definition id can't be found.
    """


class ChecksumError(Exception):
    """ Raise when there is a checksum mismatch between the
    downloaded file and source file.
    """
