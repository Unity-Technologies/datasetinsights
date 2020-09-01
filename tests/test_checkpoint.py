import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from datasetinsights.io.checkpoint import (
    EstimatorCheckpoint,
    GCSEstimatorWriter,
    LocalEstimatorWriter,
    load_from_gcs,
    load_from_http,
    load_local,
)


def create_empty_file(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    path.touch()


def test_local_estimator_writer_creates_dir():
    prefix = "good_model"
    with tempfile.TemporaryDirectory() as tmp_dir:
        dirname = os.path.join(tmp_dir, "some_dir_name")
        LocalEstimatorWriter(dirname, prefix)

        assert os.path.exists(dirname)


def test_local_writer_checkpoint_save():
    prefix = "good_model"
    suffix = "ckpt"
    estimator = Mock()

    with tempfile.TemporaryDirectory() as tmp_dir:
        dirname = os.path.join(tmp_dir, "some_dir_name")
        path = os.path.join(dirname, f"{prefix}.{suffix}")
        estimator.save = MagicMock(side_effect=create_empty_file(path))

        ckpt = LocalEstimatorWriter(dirname, prefix, suffix=suffix)
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
    url = "gs://some_bucket/path/to/directory/good_model.ckpt"
    estimator = Mock()
    mocked_ckpt = Mock()
    mocked_ckpt.save = MagicMock(return_value=path)
    mocked_gcs_client = Mock()
    mocked_gcs_client.upload = Mock()
    with patch(
        "datasetinsights.io.checkpoint.GCSClient",
        MagicMock(return_value=mocked_gcs_client),
    ):
        with patch(
            "datasetinsights.io.checkpoint.LocalEstimatorWriter",
            MagicMock(return_value=mocked_ckpt),
        ):
            gcs_ckpt = GCSEstimatorWriter(cloud_path, prefix, suffix=suffix)
            gcs_ckpt.save(estimator)

            mocked_ckpt.save.assert_called_once_with(estimator, None)
            mocked_gcs_client.upload.assert_called_once_with(
                local_path=path, url=url
            )


@pytest.mark.parametrize("filepath", ["https://some/path", "http://some/path"])
def test_get_http_loader_from_path(filepath):
    loader = EstimatorCheckpoint._get_loader_from_path(filepath)
    assert loader == load_from_http


def test_get_gcs_loader_from_path():
    loader = EstimatorCheckpoint._get_loader_from_path("gs://some/path")
    assert loader == load_from_gcs


def test_get_local_loader_from_path():
    file_name = "FasterRCNN.estimator"
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, file_name), "w") as f:
            loader = EstimatorCheckpoint._get_loader_from_path(f.name)
            assert loader == load_local


def test_get_loader_raises_error():
    filepath = "some/wrong/path"
    with pytest.raises(ValueError, match=r"Given path:"):
        EstimatorCheckpoint._get_loader_from_path(filepath)


def test_create_writer_when_checkpoint_dir_none():
    mock_local_writer = Mock()
    with patch(
        "datasetinsights.io.checkpoint.LocalEstimatorWriter",
        MagicMock(return_value=mock_local_writer),
    ):
        writer = EstimatorCheckpoint._create_writer(
            checkpoint_dir=None, estimator_name="abc"
        )

        assert writer == mock_local_writer


def test_create_writer_when_checkpoint_dir_gcs():
    mock_gcs_writer = Mock()
    with patch(
        "datasetinsights.io.checkpoint.GCSEstimatorWriter",
        MagicMock(return_value=mock_gcs_writer),
    ):
        writer = EstimatorCheckpoint._create_writer("gs://abucket/path", "def")

        assert writer == mock_gcs_writer


def test_create_writer_when_checkpoint_dir_local():
    mock_local_writer = Mock()
    with patch(
        "datasetinsights.io.checkpoint.LocalEstimatorWriter",
        MagicMock(return_value=mock_local_writer),
    ):
        with tempfile.TemporaryDirectory() as tmp:
            writer = EstimatorCheckpoint._create_writer(tmp, "abc")

            assert writer == mock_local_writer


def test_create_raises_value_error():
    incorrect_checkpoint_dir = "http://some/path"

    with pytest.raises(ValueError):
        EstimatorCheckpoint._create_writer(incorrect_checkpoint_dir, "abc")
