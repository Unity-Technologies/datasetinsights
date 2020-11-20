from unittest.mock import MagicMock, patch

import pytest
from yacs.config import CfgNode as CN

from datasetinsights.io.tracker.factory import TrackerFactory
from datasetinsights.io.tracker.mlflow import MLFlowTracker

CLIENT_ID = "test_client_id"
HOST_ID = "test_host_id"
EXP_NAME = "datasetinsights"
TEST_TOKEN = "test"
RUN_NAME = "test_run"


@pytest.fixture
def config():
    """prepare config."""
    with open("tests/configs/faster_rcnn_groceries_real_test.yaml") as f:
        cfg = CN.load_cfg(f)

    return cfg


@patch("datasetinsights.io.tracker.mlflow.id_token.fetch_id_token")
def test_refresh_token(mock_id_token):
    tracker = MagicMock()
    tracker.client_id = MagicMock(CLIENT_ID)
    mock_id_token.return_value = TEST_TOKEN
    MLFlowTracker.refresh_token(CLIENT_ID)
    mock_id_token.assert_called_once()


def test_update_run_name(config):
    config.tracker.mlflow.run = RUN_NAME
    mlflow_config = config["tracker"].get(TrackerFactory.MLFLOW_TRACKER)
    TrackerFactory.update_run_name(mlflow_config)
    assert TrackerFactory.DEFAULT_RUN_NAME == RUN_NAME


@patch("datasetinsights.io.tracker.mlflow.MLFlowTracker")
def test_get_mltracker_instance(mock_tracker):
    host_id = client_id = exp_name = "test"
    instance1 = TrackerFactory.get_mlflow_tracker_instance(
        host_id, client_id, exp_name
    )
    instance2 = TrackerFactory.get_mlflow_tracker_instance(
        host_id, client_id, exp_name
    )
    assert instance1 == instance2


@patch("datasetinsights.io.tracker.factory.NullTracker")
def test_get_nulltracker_instance(mock_tracker):
    instance1 = TrackerFactory.get_null_tracker()
    instance2 = TrackerFactory.get_null_tracker()
    assert instance1 == instance2


@patch(
    "datasetinsights.io.tracker.factory.TrackerFactory."
    "get_mlflow_tracker_instance"
)
@patch("datasetinsights.io.tracker.factory.TrackerFactory.update_run_name")
def test_factory_create_mltracker(mock_update, mock_get_tracker, config):
    config.tracker.mlflow.client_id = CLIENT_ID
    config.tracker.mlflow.host = HOST_ID
    config.tracker.mlflow.experiment = EXP_NAME
    TrackerFactory.create(config, TrackerFactory.MLFLOW_TRACKER)
    mock_get_tracker.assert_called_with(
        client_id=CLIENT_ID, exp_name=EXP_NAME, host_id=HOST_ID
    )
    mlflow_config = config["tracker"].get(TrackerFactory.MLFLOW_TRACKER)
    mock_update.assert_called_with(mlflow_config)


@patch("datasetinsights.io.tracker.factory.TrackerFactory.get_null_tracker")
def test_factory_create_nulltracker(mock_get_tracker, config):
    config.tracker.mlflow.client_id = None
    config.tracker.mlflow.host = None
    config.tracker.mlflow.experiment = None
    TrackerFactory.create(config, TrackerFactory.MLFLOW_TRACKER)
    mock_get_tracker.assert_called_with()
