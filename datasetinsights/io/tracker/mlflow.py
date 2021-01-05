import logging
import os
import threading
import time

import mlflow
from google.auth.transport.requests import Request
from google.oauth2 import id_token

from datasetinsights.constants import TIMESTAMP_SUFFIX

logger = logging.getLogger(__name__)


class MLFlowTracker:
    """ MlFlow tracker class, responsible for setting host, client_id and return
        initialized mlflow. It also refreshes the access token through daemon
        thread.
    Examples:
         # Set MLTracking server UI, default is local file
        >>> mlflow.set_tracking_uri(TRACKING_URI)
        # New run is launched under the current experiment
        >>> mlflow.start_run()
        # Log a parameter (key-value pair)
        >>> mlflow.log_param("param_name", "param_value")
        # Log a metric (key-value pair)
        >>> mlflow.log_metric("metric_name", "metric_val")
        # Log an artifact (output file)
        >>> with open("output.txt", "w") as f:
        >>>     f.write("Hello world!")
        >>> mlflow.log_artifact("output.txt", "run1/output/")
        # ends the run launched under the current experiment
        >>> mlflow.end_run()
    Attributes:
        REFRESH_INTERVAL: default refresh token interval
        __mlflow: holds initialized mlflow
    """

    REFRESH_INTERVAL = (
        3000  # every 50 minutes refresh token. Token expires in 1 hour
    )
    __mlflow = None
    CLIENT_ID = "client_id"
    HOST_ID = "host"
    EXP_NAME = "experiment"
    RUN_NAME = "run"
    DEFAULT_RUN_NAME = "run-" + TIMESTAMP_SUFFIX
    TRACKER = "tracker"
    MLFLOW_TRACKER = "mlflow"
    DEFAULT_EXP_NAME = "datasetinsights"

    def __init__(self, config):
        """constructor.
        Args:
            config : config object, holds run details
        """
        client_id, host_id, run_name, exp_name = MLFlowTracker._get_variables(
            config
        )

        if client_id:
            _refresh_token(client_id)
            thread = RefreshTokenThread(client_id)
            thread.daemon = True
            thread.start()
        mlflow.set_tracking_uri(host_id)
        mlflow.set_experiment(experiment_name=exp_name)
        logger.info(f"setting mlflow experiment name: {exp_name}")

        self.__mlflow = mlflow
        self.__mlflow.start_run(run_name=run_name)
        logger.info("instantiated mlflow")

    def get_mlflow(self):
        """ method to access initialized mlflow
        Returns:
            Initialized __mlflow instance.
        """
        logger.info("get mlflow")
        return self.__mlflow

    @staticmethod
    def _get_variables(config):
        """initialize mlflow variables.
        Args:
            config : config object, holds run details
        Returns:
            client_id: MLFlow tracking server client id
            host_id: MLFlow tracking server host id
            run_name: run name
            exp_name: experiment name
        """
        client_id = os.environ.get("MLFLOW_CLIENT_ID", None)
        host_id = os.environ.get("MLFLOW_HOST_ID", None)
        run_name = MLFlowTracker.DEFAULT_RUN_NAME
        exp_name = MLFlowTracker.DEFAULT_EXP_NAME
        tracker = config.get(MLFlowTracker.TRACKER, None)
        logger.debug(
            f"client_id:{client_id} and host_id: {host_id} from "
            f"kubernetes env variable"
        )
        if tracker and tracker.get(MLFlowTracker.MLFLOW_TRACKER, None):
            mlflow_config = tracker.get(MLFlowTracker.MLFLOW_TRACKER)
            host_id = mlflow_config.get(MLFlowTracker.HOST_ID, host_id)
            client_id = mlflow_config.get(MLFlowTracker.CLIENT_ID, client_id)
            run_name = mlflow_config.get(MLFlowTracker.RUN_NAME, run_name)
            exp_name = mlflow_config.get(MLFlowTracker.EXP_NAME, exp_name)
            logger.debug(
                f"client_id:{client_id} and host_id:{host_id} from yaml config"
            )
        logger.info(
            f"client_id:{client_id} and host_id:{host_id} connecting to mlflow"
        )
        if not host_id:
            logger.warning(f"host_id not found")
            raise ValueError("host_id not configured")
        return client_id, host_id, run_name, exp_name


class RefreshTokenThread(threading.Thread):
    """ Its service thread which keeps running till main thread runs
        and refresh access tokens.
    Attributes:
        client_id: MLFlow tracking server client id
        interval: duration at which it refreshes the token
    """

    def __init__(self, client_id, interval=MLFlowTracker.REFRESH_INTERVAL):
        """constructor.
        Args:
            client_id : MLFlow tracking server client id
            interval: duration at which it refreshes the token
        """
        threading.Thread.__init__(self)
        self.client_id = client_id
        self.interval = interval

    def run(self):
        """ Thread run method which keeps running at specified interval
            till main thread runs.
        """
        while True:
            _refresh_token(self.client_id)
            logger.info(
                f"RefreshTokenThread: updated token, sleeping for "
                f"{self.interval} seconds"
            )
            time.sleep(self.interval)


def _refresh_token(client_id):
    """refresh token and set in environment variable.
    Args:
        client_id : MLFlow tracking server client id
    """
    if client_id:
        google_open_id_connect_token = id_token.fetch_id_token(
            Request(), client_id
        )
        os.environ["MLFLOW_TRACKING_TOKEN"] = google_open_id_connect_token
        logger.info("refreshing o-auth token for mlflow")
