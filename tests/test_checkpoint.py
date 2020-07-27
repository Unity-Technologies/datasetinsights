import os
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

import datasetinsights.constants as const
from datasetinsights.storage.checkpoint import (
    EstimatorCheckpoint,
    GCSEstimatorWriter,
    LocalEstimatorWriter,
)


def create_empty_file(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    path.touch()


def test_local_estimator_writer_creates_dir():
    prefix = "good_model"
    dirname = os.path.join(const.PROJECT_ROOT, prefix)
    LocalEstimatorWriter(prefix)
    assert os.path.exists(dirname)


def test_local_writer_checkpoint_save():
    prefix = "good_model"
    suffix = "ckpt"
    estimator = Mock()

    dirname = os.path.join(const.PROJECT_ROOT, prefix)
    path = os.path.join(dirname, f"{prefix}.{suffix}")
    estimator.save = MagicMock(side_effect=create_empty_file(path))

    ckpt = LocalEstimatorWriter(prefix, suffix=suffix)
    ckpt.save(estimator)

    assert os.path.exists(path)

    epoch = 12
    path_with_epoch = os.path.join(dirname, f"{prefix}.ep{epoch}.{suffix}")
    estimator.save = MagicMock(side_effect=create_empty_file(path_with_epoch))

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
            "datasetinsights.storage.checkpoint.LocalEstimatorWriter",
            MagicMock(return_value=mocked_ckpt),
        ):
            gcs_ckpt = GCSEstimatorWriter(cloud_path, prefix, suffix=suffix)
            gcs_ckpt.save(estimator)

            mocked_ckpt.save.assert_called_once_with(estimator, None)
            mocked_gcs_client.upload.assert_called_once_with(
                path, bucket, object_key
            )


def test_create_writer():
    mock_local_writer = Mock()
    with patch(
        "datasetinsights.storage.checkpoint.LocalEstimatorWriter",
        MagicMock(return_value=mock_local_writer),
    ):
        writer = EstimatorCheckpoint._create_writer("/path/to/folder", "abc")

        assert writer == mock_local_writer

    mock_gcs_writer = Mock()
    with patch(
        "datasetinsights.storage.checkpoint.GCSEstimatorWriter",
        MagicMock(return_value=mock_gcs_writer),
    ):
        writer = EstimatorCheckpoint._create_writer("gs://abucket/path", "def")

        assert writer == mock_gcs_writer


@patch("datasetinsights.storage.checkpoint.load_from_http")
@patch("datasetinsights.storage.checkpoint.load_from_gcs")
@patch("datasetinsights.storage.checkpoint.load_local")
def test_get_loader_from_path(mock_local, mock_gcs, mock_http):
    loader = EstimatorCheckpoint._get_loader_from_path("gs://some/path")
    assert loader == mock_gcs

    loader = EstimatorCheckpoint._get_loader_from_path("http://some/path")
    assert loader == mock_http

    loader = EstimatorCheckpoint._get_loader_from_path("https://some/path")
    assert loader == mock_http

    loader = EstimatorCheckpoint._get_loader_from_path("/path/to/folder")
    assert loader == mock_local

    with pytest.raises(ValueError, match=r"Given path:"):
        EstimatorCheckpoint._get_loader_from_path("dfdge")
