""" Save estimator checkpoints
"""
import logging
import os
import tempfile

import datasetinsights.constants as const
from datasetinsights.io.download import download_file
from datasetinsights.io.gcs import GCSClient
from datasetinsights.torch_distributed import is_master

logger = logging.getLogger(__name__)

DEFAULT_SUFFIX = "estimator"


class EstimatorCheckpoint:
    """Saves and loads estimator checkpoints.

    Assigns estimator checkpoint writer according to log_dir which is
    responsible for saving estimators. Writer can be a GCS or local writer.
    Assigns loader which is responsible for loading estimator from a given
    path. Loader can a local, GCS or HTTP loader.

    Args:
        estimator_name (str): name of the estimator
        checkpoint_dir (str): Directory where checkpoints are stored
        distributed (bool): boolean to determine distributed training

    Attributes:
        checkpoint_dir (str): Directory where checkpoints are stored
        distributed (bool): boolean to determine distributed training

    """

    def __init__(self, estimator_name, checkpoint_dir, distributed):
        self.distributed = distributed
        self._writer = self._create_writer(checkpoint_dir, estimator_name)

    @staticmethod
    def _create_writer(checkpoint_dir, estimator_name):
        """Creates writer object for saving checkpoints based on log_dir.

        Args:
            checkpoint_dir: Directory where checkpoints are stored
            estimator_name: Name of the estimator.

        Returns:
            Writer object (GCS or Local).
        """

        if checkpoint_dir is None:
            checkpoint_dir = const.DEFAULT_CHECKPOINT_DIR
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

        if checkpoint_dir.startswith(const.GCS_BASE_STR):
            writer = GCSEstimatorWriter(checkpoint_dir, estimator_name)
        elif os.path.isdir(checkpoint_dir):
            writer = LocalEstimatorWriter(checkpoint_dir, estimator_name)
        else:
            raise ValueError(
                f"Can not use {checkpoint_dir} as checkpoint directory."
            )

        return writer

    @staticmethod
    def _get_loader_from_path(path):
        """Gives loader method to load an estimator from a given path.

        Arguments:
            path: Path of the estimator. Can be local, GCS or HTTP path.

        Returns:
            Loader method for loading estimator according to the given path.
        """
        if path.startswith((const.HTTP_URL_BASE_STR, const.HTTPS_URL_BASE_STR)):
            method = load_from_http
        elif path.startswith(const.GCS_BASE_STR):
            method = load_from_gcs
        elif os.path.isfile(path):
            method = load_local
        else:
            raise ValueError(
                f"Given path: {path}, is either invalid or not supported."
                f"Currently supported path are local path (/path/to/file), "
                f"GCS path (gs://) and HTTP url (http:// or https://) path"
            )

        return method

    def save(self, estimator, epoch):
        """Save estimator to the log_dir.

        Args:
            estimator (datasetinsights.estimators.Estimator):
            datasetinsights estimator object.
            epoch (int): Epoch number.

        """
        if self.distributed and not is_master():
            return
        self._writer.save(estimator=estimator, epoch=epoch)

    def load(self, estimator, path):
        """Loads estimator from given path.

        Path can be either a local path or GCS path or HTTP url.

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
    """Writes (saves) estimator checkpoints locally.

    Args:
        dirname (str): Directory where estimator is to be saved.
        prefix (str): Filename prefix of the checkpoint files.
        suffix (str): Filename suffix of the checkpoint files.
        create_dir (bool): Flag for creating new directory. Default: True.

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
        """Save estimator to locally to log_dir.

        Args:
            estimator (datasetinsights.estimators.Estimator):
                datasetinsights estimator object.
            epoch (int): The current epoch number. Default: None

        Returns:
            Full path to the saved checkpoint file.

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
    """Writes (saves) estimator checkpoints on GCS.

    Args:
        cloud_path (str): GCS cloud path (e.g. gs://bucket/path/to/directoy)
        prefix (str): filename prefix of the checkpoint files
        suffix (str): filename suffix of the checkpoint files

    """

    def __init__(self, cloud_path, prefix, *, suffix=DEFAULT_SUFFIX):
        self._tempdir = tempfile.TemporaryDirectory().name
        self._client = GCSClient()
        self.cloud_path = cloud_path
        self._writer = LocalEstimatorWriter(
            self._tempdir, prefix, create_dir=True, suffix=suffix
        )

    def save(self, estimator, epoch=None):
        """Save estimator to checkpoint files on GCS.

        Args:
            estimator (datasetinsights.estimators.Estimator):
                datasetinsights estimator object.
            epoch (int): the current epoch number. Default: None

        Returns:
            Full GCS cloud path to the saved checkpoint file.

        """
        path = self._writer.save(estimator, epoch)
        filename = os.path.basename(path)
        full_cloud_path = os.path.join(self.cloud_path, filename)
        logger.debug(f"Copying estimator from {path} to {full_cloud_path}")
        self._client.upload(local_path=path, url=full_cloud_path)
        return full_cloud_path


def load_local(estimator, path):
    """Loads estimator checkpoints from a local path."""
    estimator.load(path)

    return path


def load_from_gcs(estimator, full_cloud_path):
    """Load estimator from checkpoint files on GCS.

    Args:
        estimator (datasetinsights.estimators.Estimator):
            datasetinsights estimator object.
        full_cloud_path: full path to the checkpoint file

    """
    filename = os.path.basename(full_cloud_path)
    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, filename)
        logger.debug(f"Downloading estimator from {full_cloud_path} to {path}")
        client = GCSClient()
        client.download(local_path=temp_dir, url=full_cloud_path)
        estimator.load(path)


def load_from_http(estimator, url):
    """Load estimator from checkpoint files on GCS.

    Args:
        estimator (datasetinsights.estimators.Estimator):
            datasetinsights estimator object.
        url: URL of the checkpoint file

    """
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.debug(f"Downloading estimator from {url} to {temp_dir}")
        path = download_file(source_uri=url, dest_path=temp_dir)
        logger.debug(f"Loading estimator from {path}")
        estimator.load(path)
