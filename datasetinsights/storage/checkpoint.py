""" Save estimator checkpoints
"""
import logging
import os
import tempfile

import datasetinsights.constants as const
from datasetinsights.data.download import download_file
from datasetinsights.torch_distributed import is_master

from .gcs import GCSClient, gcs_bucket_and_path

logger = logging.getLogger(__name__)

DEFAULT_SUFFIX = "estimator"


class EstimatorCheckpoint:
    """
    For loading and saving estimator checkpoints

    Args:
        estimator_name (str): name of the estimator
        log_dir (str): log directory
        distributed (bool): boolean to determine distributed training
    Attributes:
        estimator_name (str): name of the estimator
        log_dir (str): log directory
        distributed (bool): boolean to determine distributed training
    """

    def __init__(self, estimator_name, log_dir, distributed):
        self.distributed = distributed
        self._writer = self._create_writer(log_dir, estimator_name)

    @staticmethod
    def _create_writer(log_dir, estimator_name):
        if log_dir.startswith(const.GCS_BASE_STR):
            writer = GCSEstimatorWriter(log_dir, estimator_name)
        else:
            writer = LocalEstimatorWriter(estimator_name)

        return writer

    @staticmethod
    def _get_loader_from_path(path):
        if path.startswith((const.HTTP_URL_BASE_STR, const.HTTPS_URL_BASE_STR)):
            method = load_from_http
        elif path.startswith(const.GCS_BASE_STR):
            method = load_from_gcs
        elif path.startswith("/"):
            method = load_local
        else:
            raise ValueError(
                f"Given path: {0}, is either invalid or not "
                "supported".format(path)
            )

        return method

    def save(self, estimator, epoch):
        """
        Saves estimator to log directory

        Args:
            estimator (datasetinsights.estimators.Estimator):
            datasetinsights estimator object
            epoch (int): epoch number
        """
        if self.distributed and not is_master():
            return
        self._writer.save(estimator=estimator, epoch=epoch)

    def load(self, estimator, path):
        """ Loads estimator from given path

        Path can be either a local path or GCS path or HTTP url

        Args:
            estimator (datasetinsights.estimators.Estimator):
            datasetinsights estimator object
            path (str): path of estimator
        """
        if self.distributed and not is_master():
            return

        load_method = self._get_loader_from_path(path)
        load_method(estimator, path)


class LocalEstimatorWriter:
    """ Writes (saves) estimator checkpoints locally

    Args:
        prefix (str): filename prefix of the checkpoint files
        suffix (str): filename suffix of the checkpoint files

    Attributes:
        dirname (str): directory name of where checkpoint files are stored
        prefix (str): filename prefix of the checkpoint files
        suffix (str): filename suffix of the checkpoint files
    """

    def __init__(self, prefix, *, suffix=DEFAULT_SUFFIX):
        self.dirname = os.path.join(const.PROJECT_ROOT, prefix)
        self.prefix = prefix
        self.suffix = suffix

        if not os.path.exists(self.dirname):
            os.makedirs(self.dirname)

    def save(self, estimator, epoch=None):
        """ Save estimator to checkpoint files.

        Args:
            estimator (datasetinsights.estimators.Estimator):
                datasetinsights estimator object.
            epoch (int): the current epoch number. Default: None

        Returns:
            full path to the saved checkpoint file
        """
        if epoch:
            filename = ".".join([self.prefix, f"ep{epoch}", self.suffix])
        else:
            filename = ".".join([self.prefix, self.suffix])
        path = os.path.join(self.dirname, filename)

        logger.debug(f"Saving estimator to {path}")
        estimator.save(path)

        return path


class GCSEstimatorWriter:
    """ Writes (saves) estimator checkpoints on GCS

    Args:
        cloud_path (str): GCS cloud path (e.g. gs://bucket/path/to/directoy)
        prefix (str): filename prefix of the checkpoint files
        suffix (str): filename suffix of the checkpoint files

    """

    def __init__(self, cloud_path, prefix, *, suffix=DEFAULT_SUFFIX):
        self._client = GCSClient()
        self._bucket, self._gcs_path = gcs_bucket_and_path(cloud_path)
        self._writer = LocalEstimatorWriter(prefix, suffix=suffix)

    def save(self, estimator, epoch=None):
        """ Save estimator to checkpoint files on GCS

        Args:
            estimator (datasetinsights.estimators.Estimator):
                datasetinsights estimator object.
            epoch (int): the current epoch number. Default: None

        Returns:
            full GCS cloud path to the saved checkpoint file
        """
        path = self._writer.save(estimator, epoch)
        filename = os.path.basename(path)
        object_key = os.path.join(self._gcs_path, filename)

        full_cloud_path = f"gs://{self._bucket}/{object_key}"

        logger.debug(f"Copying estimator from {path} to {full_cloud_path}")
        self._client.upload(path, self._bucket, object_key)

        return full_cloud_path


def load_local(estimator, path):
    """ loads estimator checkpoints from a local path """
    estimator.load(path)

    return path


def load_from_gcs(estimator, full_cloud_path):
    """ Load estimator from checkpoint files on GCS

    Args:
        estimator (datasetinsights.estimators.Estimator):
            datasetinsights estimator object.
        full_cloud_path: full path to the checkpoint file
    """
    bucket, object_key = gcs_bucket_and_path(full_cloud_path)
    filename = os.path.basename(object_key)
    temp_dir = tempfile.TemporaryDirectory().name
    path = os.path.join(temp_dir, filename)
    logger.debug(f"Downloading estimator from {full_cloud_path} to {path}")
    client = GCSClient()
    client.download(bucket, object_key, path)
    estimator.load(estimator, path)

    return path


def load_from_http(estimator, url):
    """ Load estimator from checkpoint files on GCS

    Args:
        estimator (datasetinsights.estimators.Estimator):
            datasetinsights estimator object.
        url: URL of the checkpoint file
    """
    temp_dir = tempfile.TemporaryDirectory().name
    path = os.path.join(temp_dir, "filename")
    logger.debug(f"Downloading estimator from {url} to {path}")
    download_file(source_uri=url, dest_path=path)
    logger.debug(f"Loading estimator from {path}")
    estimator.load(path)

    return path
