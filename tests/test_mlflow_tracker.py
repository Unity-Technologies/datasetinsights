from unittest.mock import MagicMock, patch

import pytest
from yacs.config import CfgNode as CN

from datasetinsights.constants import MLFLOW_TRACKER
from datasetinsights.io.tracker.factory import TrackerFactory
from datasetinsights.io.tracker.mzflow import MLFlowTracker

CLIENT_ID = "test_client_id"
HOST_ID = "test_host_id"
EXP_NAME = "datasetinsights"
TEST_TOKEN = "test"


@pytest.fixture
def config():
    """prepare config."""
    with open("tests/configs/faster_rcnn_groceries_real_test.yaml") as f:
        cfg = CN.load_cfg(f)

    return cfg


def test_get_instance():
    first_instance = MLFlowTracker.get_instance()
    second_instance = MLFlowTracker.get_instance()
    assert first_instance == second_instance


@patch("datasetinsights.io.tracker.mzflow.mlflow")
@patch("datasetinsights.io.tracker.mzflow.MLFlowTracker.refresh_token")
def test_get_tracker(mock_refresh, mock_mlflow, config):

    config.mlflow.tracking_client_id = CLIENT_ID
    config.mlflow.tracking_host_id = HOST_ID
    config.mlflow.experiment_name = EXP_NAME
    MLFlowTracker.get_tracker(config)
    mock_refresh.assert_called_with(CLIENT_ID)
    mock_mlflow.set_tracking_uri.assert_called_with(HOST_ID)
    mock_mlflow.set_experiment.assert_called_with(experiment_name=EXP_NAME)
    assert not config.get("mlflow")


@patch("datasetinsights.io.tracker.mzflow.id_token.fetch_id_token")
@patch("datasetinsights.io.tracker.mzflow.MLFlowTracker.get_instance")
def test_refresh_token(mock_get_instance, mock_id_token):
    tracker = MagicMock()
    tracker.client_id = MagicMock(CLIENT_ID)
    mock_get_instance.return_value = tracker
    mock_id_token.return_value = TEST_TOKEN
    MLFlowTracker.refresh_token(CLIENT_ID)
    mock_get_instance.assert_called_with(CLIENT_ID)


@patch("datasetinsights.io.tracker.mzflow.MLFlowTracker.get_tracker")
def test_create_tracker_factory(mock_get_tracker, config):
    TrackerFactory.create(config, MLFLOW_TRACKER)
    mock_get_tracker.assert_called_with(config)
