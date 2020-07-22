""" Save estimator checkpoints
"""
import logging
import os
import tempfile

import datasetinsights.constants as const
from datasetinsights.torch_distributed import is_master

from .gcs import GCSClient, gcs_bucket_and_path

logger = logging.getLogger(__name__)

DEFAULT_SUFFIX = "estimator"
GCS_BASE_STR = "gs://"


def checkpoint_file_on_gcs(config):
    """
    determine whether or not the checkpoint file to load an estimator is on
    gcs

    Args:
        config:

    Returns:

    """
    return (
        "checkpoint_file" in config.keys()
        and config.checkpoint_file != const.NULL_STRING
        and config.checkpoint_file[: len(GCS_BASE_STR)] == GCS_BASE_STR
    )


def create_checkpointer(*, logdir, config):
    """
    Initialize the correct estimator checkpointer
    Args:
        logdir: filepath to where to save/load local copy of estimator
        config:

    Returns (EstimatorCheckpoint): the correct estimator checkpoint for the
    config. The Estimator Checkpoint is responsible for saving and loading
    the estimator.

    """
    if logdir.startswith(GCS_BASE_STR):
        checkpointer = GCSEstimatorCheckpoint(logdir, config.estimator)
    elif checkpoint_file_on_gcs(config):
        logdir = f"{GCS_BASE_STR}{const.GCS_BUCKET}/runs/{str(logdir)}"
        checkpointer = GCSEstimatorCheckpoint(logdir, config.estimator)
    else:
        checkpointer = EstimatorCheckpoint(logdir, config.estimator)
    if config.system.distributed:
        checkpointer = DistributedEstimatorCheckpoint(
            is_master=is_master(), estimator_checkpoint=checkpointer
        )
    return checkpointer


class EstimatorCheckpoint:
    """ Interact with estimator checkpoints

    Args:
        dirname (str): directory name of where checkpoint files are stored
        prefix (str): filename prefix of the checkpoint files
        suffix (str): filename suffix of the checkpoint files
        create_dir (bool): indicate whether to force create directoy if
            the specified dirname does not exist. Default: True

    Attributes:
        dirname (str): directory name of where checkpoint files are stored
        prefix (str): filename prefix of the checkpoint files
        suffix (str): filename suffix of the checkpoint files
    """

    def __init__(
        self, dirname, prefix, *, suffix=DEFAULT_SUFFIX, create_dir=True
    ):
        self.dirname = dirname
        self.prefix = prefix
        self.suffix = suffix
        self.is_master = is_master()
        if create_dir:
            if not os.path.exists(dirname):
                os.makedirs(dirname)

        if not os.path.exists(dirname):
            raise ValueError(f"Directory path '{dirname}' is not found.")

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

    def load(self, estimator, path):
        """ Load estimator from checkpoint files

        Args:
            estimator (datasetinsights.estimators.Estimator):
                datasetinsights estimator object.
            path: full path to the checkpoint file
        """
        logger.debug(f"Loading estimator from {path}")
        estimator.load(path)


class GCSEstimatorCheckpoint:
    """ Interact with estimator checkpoints on GCS

    Args:
        cloud_path (str): GCS cloud path (e.g. gs://bucket/path/to/directoy)
        prefix (str): filename prefix of the checkpoint files
        suffix (str): filename suffix of the checkpoint files
    """

    def __init__(self, cloud_path, prefix, *, suffix=DEFAULT_SUFFIX):
        self._tempdir = tempfile.TemporaryDirectory().name
        self._client = GCSClient()
        self._bucket, self._gcs_path = gcs_bucket_and_path(cloud_path)
        self._checkpointer = EstimatorCheckpoint(
            self._tempdir, prefix, create_dir=True, suffix=suffix
        )

    def save(self, estimator, epoch=None):
        """ Save estimator to checkpoint files on GCS

        Args:
            estimator (datasetinsights.estimators.Estimator):
                datasetinsights estimator object.
            epoch (int): the current epoch number. Default: None

        Returns:
            full GCS cloud path to the saved checkpoint file
        """
        path = self._checkpointer.save(estimator, epoch)
        filename = os.path.basename(path)
        object_key = os.path.join(self._gcs_path, filename)

        full_cloud_path = f"gs://{self._bucket}/{object_key}"

        logger.debug(f"Copying estimator from {path} to {full_cloud_path}")
        self._client.upload(path, self._bucket, object_key)

        return full_cloud_path

    def load(self, estimator, full_cloud_path):
        """ Load estimator from checkpoint files on GCS

        Args:
            estimator (datasetinsights.estimators.Estimator):
                datasetinsights estimator object.
            path: full path to the checkpoint file
        """
        bucket, object_key = gcs_bucket_and_path(full_cloud_path)
        filename = os.path.basename(object_key)
        path = os.path.join(self._tempdir, filename)
        logger.debug(f"Downloading estimator from {full_cloud_path} to {path}")
        self._client.download(bucket, object_key, path)

        self._checkpointer.load(estimator, path)


class DistributedEstimatorCheckpoint:
    def __init__(self, is_master, estimator_checkpoint):
        self.is_master = is_master
        self.estimator_checkpoint = estimator_checkpoint

    def save(self, estimator, epoch=None):
        if self.is_master:
            self.estimator_checkpoint.save(estimator, epoch)

    def load(self, estimator, full_cloud_path):
        if self.is_master:
            self.estimator_checkpoint.load(estimator, full_cloud_path)
