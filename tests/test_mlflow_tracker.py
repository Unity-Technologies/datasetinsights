from unittest.mock import MagicMock, patch

import pytest
from yacs.config import CfgNode as CN

from datasetinsights.io.tracker.factory import NullTracker, TrackerFactory
from datasetinsights.io.tracker.mlflow import MLFlowTracker, RefreshTokenThread

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
    tf = TrackerFactory()
    instance1 = tf._mlflow_tracker_instance(host_id, client_id, exp_name)
    instance2 = tf._mlflow_tracker_instance(host_id, client_id, exp_name)
    assert instance1 == instance2


@patch("datasetinsights.io.tracker.factory.NullTracker")
def test_get_nulltracker_instance(mock_tracker):
    tf = TrackerFactory()
    instance1 = tf._null_tracker()
    instance2 = tf._null_tracker()
    assert instance1 == instance2


@patch(
    "datasetinsights.io.tracker.factory.TrackerFactory."
    "_mlflow_tracker_instance"
)
@patch("datasetinsights.io.tracker.factory.TrackerFactory.update_run_name")
def test_factory_create_mltracker(mock_update, mock_get_tracker, config):
    mock_mlflow = MagicMock()
    mock_get_tracker.return_value = mock_mlflow

    mock_mlflowtracker = mock_mlflow.get_mlflow()
    config.tracker.mlflow.client_id = CLIENT_ID
    config.tracker.mlflow.host = HOST_ID
    config.tracker.mlflow.experiment = EXP_NAME
    TrackerFactory.create(config, TrackerFactory.MLFLOW_TRACKER)
    mock_get_tracker.assert_called_with(
        client_id=CLIENT_ID, exp_name=EXP_NAME, host_id=HOST_ID
    )
    mlflow_config = config["tracker"].get(TrackerFactory.MLFLOW_TRACKER)
    mock_update.assert_called_with(mlflow_config)
    mock_mlflow.get_mlflow.assert_called_with()
    mock_mlflowtracker.start_run.assert_called_with(run_name=RUN_NAME)


@patch("datasetinsights.io.tracker.factory.TrackerFactory._null_tracker")
def test_factory_create_nulltracker(mock_get_tracker, config):
    config.tracker.mlflow.client_id = None
    config.tracker.mlflow.host = None
    config.tracker.mlflow.experiment = None
    TrackerFactory.create(config, TrackerFactory.MLFLOW_TRACKER)
    mock_get_tracker.assert_called_with()


@patch("datasetinsights.io.tracker.factory.NullTracker.handle_dummy")
def test__nulltracker(mock_handle_dummy):
    null_tracker = NullTracker()
    null_tracker.start_run(run_name=RUN_NAME)
    mock_handle_dummy.assert_called_with(run_name=RUN_NAME)


@patch("datasetinsights.io.tracker.mlflow.RefreshTokenThread.start")
@patch("datasetinsights.io.tracker.mlflow.mlflow")
@patch("datasetinsights.io.tracker.mlflow.MLFlowTracker.refresh_token")
def test__mLflow_tracker(mock_refresh, mock_mlflow, mock_thread_start, config):
    config.tracker.mlflow.client_id = CLIENT_ID
    config.tracker.mlflow.host = HOST_ID
    config.tracker.mlflow.experiment = EXP_NAME
    MLFlowTracker(HOST_ID, CLIENT_ID, EXP_NAME)
    mock_thread_start.assert_called_with()
    mock_refresh.assert_called_once()
    mock_mlflow.set_tracking_uri.assert_called_with(HOST_ID)


@patch("datasetinsights.io.tracker.mlflow.RefreshTokenThread.run")
def test__refresh_token_thread(mock_thread_run):
    thread = RefreshTokenThread(CLIENT_ID)
    thread.daemon = True
    thread.start()
    mock_thread_run.assert_called_once()
