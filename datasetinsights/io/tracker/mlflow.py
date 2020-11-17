import os
import threading
import time

import mlflow
from google.auth.transport.requests import Request
from google.oauth2 import id_token


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
        __instance: holds singleton instance
        __mlflow: holds initialized mlflow

    """

    REFRESH_INTERVAL = 480  # 8 minute refresh intervals
    __instance = None
    __mlflow = None

    def __init__(self, host_id, client_id, exp_name):
        """constructor.

        Args:
            host_id: MlTracker server host
            client_id: MLFlow tracking server client id
            exp_name: name of the experiment
        """
        if MLFlowTracker.__instance:
            raise Exception("This class is a singleton!")
        else:
            if client_id:
                MLFlowTracker.refresh_token(client_id)
                thread = BackgroundThread(client_id)
                thread.daemon = True
                thread.start()
            mlflow.set_tracking_uri(host_id)
            if exp_name:
                mlflow.set_experiment(experiment_name=exp_name)
            self.__mlflow = mlflow
            MLFlowTracker.__instance = self

    @staticmethod
    def get_instance():

        """Static instance access method.
        Returns:
            MLFlowtracker singleton instance.
        """
        return MLFlowTracker.__instance

    def get_mlflow(self):

        """ method to access initialized mlflow
        Returns:
            Initialized __mlflow instance.
        """
        return self.__mlflow

    @staticmethod
    def refresh_token(client_id):
        """refresh token and set in environment variable.

        Args:
            client_id : MLFlow tracking server client id
        """
        if client_id:
            google_open_id_connect_token = id_token.fetch_id_token(
                Request(), client_id
            )
            os.environ["MLFLOW_TRACKING_TOKEN"] = google_open_id_connect_token


class BackgroundThread(threading.Thread):
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
            MLFlowTracker.refresh_token(self.client_id)
            time.sleep(self.interval)
