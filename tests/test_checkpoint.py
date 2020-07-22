import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from datasetinsights.storage.checkpoint import (
    EstimatorCheckpoint,
    GCSEstimatorCheckpoint,
)


def create_empty_file(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    path.touch()


def test_estimator_checkpoint_creates_dir():
    prefix = "good_model"
    with tempfile.TemporaryDirectory() as tmp_dir:
        dirname = os.path.join(tmp_dir, "some_dir_name")
        EstimatorCheckpoint(dirname, prefix)

        assert os.path.exists(dirname)


def test_estimator_checkpoint_raises():
    prefix = "good_model"
    with tempfile.TemporaryDirectory() as tmp_dir:
        dirname = os.path.join(tmp_dir, "some_dir_name")
        with pytest.raises(ValueError, match=r"Directory path"):
            EstimatorCheckpoint(dirname, prefix, create_dir=False)


def test_estimator_checkpoint_save():
    prefix = "good_model"
    suffix = "ckpt"
    estimator = Mock()

    with tempfile.TemporaryDirectory() as tmp_dir:
        dirname = os.path.join(tmp_dir, "some_dir_name")
        path = os.path.join(dirname, f"{prefix}.{suffix}")
        estimator.save = MagicMock(side_effect=create_empty_file(path))

        ckpt = EstimatorCheckpoint(dirname, prefix, suffix=suffix)
        ckpt.save(estimator)

        assert os.path.exists(path)

        epoch = 12
        path_with_epoch = os.path.join(dirname, f"{prefix}.ep{epoch}.{suffix}")
        estimator.save = MagicMock(
            side_effect=create_empty_file(path_with_epoch)
        )

        ckpt.save(estimator, epoch)

        assert os.path.exists(path_with_epoch)


def test_gcs_estimator_checkpoint_save():
    bucket = "some_bucket"
    gcs_path = "path/to/directory"
    cloud_path = f"gs://{bucket}/{gcs_path}"
    prefix = "good_model"
    suffix = "ckpt"
    path = "/path/does/not/matter/" + ".".join([prefix, suffix])
    object_key = os.path.join(gcs_path, ".".join([prefix, suffix]))

    estimator = Mock()
    mocked_ckpt = Mock()
    mocked_ckpt.save = MagicMock(return_value=path)
    mocked_gcs_client = Mock()
    mocked_gcs_client.upload = Mock()
    with patch(
        "datasetinsights.storage.checkpoint.GCSClient",
        MagicMock(return_value=mocked_gcs_client),
    ):
        with patch(
            "datasetinsights.storage.checkpoint.EstimatorCheckpoint",
            MagicMock(return_value=mocked_ckpt),
        ):
            gcs_ckpt = GCSEstimatorCheckpoint(cloud_path, prefix, suffix=suffix)
            gcs_ckpt.save(estimator)

            mocked_ckpt.save.assert_called_once_with(estimator, None)
            mocked_gcs_client.upload.assert_called_once_with(
                path, bucket, object_key
            )
