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
    """ MlFlow tracker class:
        Responsible for setting host, client_id and return
        initialized mlflow. It also refreshes the access token through daemon
        thread. To start mlflow, host is required either through config YAML or
        Kubernetes secrets.
    Possible cases:
        Case 1: Host and client_id are not configured:
                Null tracker will be initiated.
        Case 2: Host and client_id both are configured:
                mlflow will initiate background thread to refresh token and
                will start mlflow tracker.
        Case 3: Only Host id is configured and client_id is None:
                mlflow tracker will start without initiating background thread.
    Order of lookup:
        If host and client_id are configured in YAML then that
        will be used else it will lookup in Kubernetes env variable.
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
    YAML_Config:
        >>> tracker:
        >>>   mlflow:
        >>>     experiment:
        >>>     run:
        >>>     client_id:
        >>>     host:
    Attributes:
        REFRESH_INTERVAL: default refresh token interval
        __mlflow: holds initialized mlflow
    """

    REFRESH_INTERVAL = (
        3000  # every 50 minutes refresh token. Token expires in 1 hour
    )
    __mlflow = None
    DEFAULT_RUN = "run-" + TIMESTAMP_SUFFIX
    DEFAULT_EXP = "datasetinsights"

    def __init__(
        self,
        *,
        client_id=None,
        host=None,
        run=DEFAULT_RUN,
        experiment=DEFAULT_EXP,
    ):
        """constructor.
        Args:
            client_id(str, optional): MLFlow tracking server client id
            host(str, optional): MLFlow tracking server host name
            run(str, optional): MLFlow tracking run name
            experiment(str, optional): MLFlow tracking experiment name
        Raises:
            ValueError: If `host_id` is not available in both YAML config
            and env variable.
        """
        host = host or os.environ.get("MLFLOW_HOST_ID", None)
        MLFlowTracker._validate(host)
        client_id = client_id or os.environ.get("MLFLOW_CLIENT_ID", None)
        logger.info(
            f"client_id:{client_id} and host_id:{host} connecting to mlflow"
        )
        if client_id:
            _refresh_token(client_id)
            thread = RefreshTokenThread(client_id)
            thread.daemon = True
            thread.start()
        mlflow.set_tracking_uri(host)
        mlflow.set_experiment(experiment_name=experiment)
        experiment_id = mlflow.get_experiment_by_name(experiment).experiment_id
        logger.info(
            f"Starting mlflow: experiment name: {experiment} "
            f"and experiment id: {experiment_id}"
        )

        self.__mlflow = mlflow
        active_run = self.__mlflow.start_run(run_name=run)
        logger.info(f"Instantiated mlflow: run id: {active_run.info.run_id}")

    def get_mlflow(self):
        """ method to access initialized mlflow
        Returns:
            Initialized __mlflow instance.
        """
        logger.info("get mlflow")
        return self.__mlflow

    @staticmethod
    def _validate(host):
        if not host:
            logger.warning(f"host_id not found")
            raise ValueError("host_id not configured")


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
