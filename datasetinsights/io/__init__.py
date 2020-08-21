import gzip
import os
import shutil

from .bbox import BBox2D
from .checkpoint import EstimatorCheckpoint
from .downloader import create_downloader
from .kfp_output import KubeflowPipelineWriter

__all__ = [
    "BBox2D",
    "EstimatorCheckpoint",
    "KubeflowPipelineWriter",
    "create_downloader",
]


def gzip_decompress(filepath, destination):
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


shutil.register_unpack_format("gz", [".gz"], gzip_decompress)
