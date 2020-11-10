import os

import mlflow
from google.auth.transport.requests import Request
from google.oauth2 import id_token

MLFLOW = "mlflow"
CLIENT_ID = "tracking_client_id"
HOST_ID = "tracking_host_id"
NAME = "experiment_name"
DEFAULT_NAME = "datasetinsights"


class MLFlowTracker:
    """MlFlow tracker class, responsible for creating singleton object,
    refresh token and get instance .

    Attributes:
        __instance: holds singleton instance
        __client_id: holds ml tracking server client id

    """

    __instance = None
    __client_id = None

    def __init__(self, client_id):
        """Virtually private constructor.

        Args:
            client_id : MLFlow tracking server client id
        """
        if MLFlowTracker.__instance:
            raise Exception("This class is a singleton!")
        else:
            if client_id:
                self.__client_id = client_id
            MLFlowTracker.__instance = self

    @staticmethod
    def get_instance(client_id=None):

        """Static instance access method.

        Args:
            client_id : MLFlow tracking server client id
        """

        if not MLFlowTracker.__instance:
            MLFlowTracker(client_id)
        return MLFlowTracker.__instance

    @staticmethod
    def get_tracker(config):

        """get instance of tracker class.

        Args:
            config : config object, holds server details
        """
        client_id = config.get(MLFLOW).get(CLIENT_ID, None)
        host_id = config.get(MLFLOW).get(HOST_ID, None)
        exp_name = config.get(MLFLOW).get(NAME, DEFAULT_NAME)
        if host_id:
            if client_id:
                MLFlowTracker.refresh_token(client_id)
            mlflow.set_tracking_uri(host_id)
            mlflow.set_experiment(experiment_name=exp_name)
        config.pop(MLFLOW, None)
        return mlflow

    @staticmethod
    def refresh_token(client_id=None):
        """refresh token and set in environment variable.

        Args:
            client_id : MLFlow tracking server client id
        """
        ml_tracker = MLFlowTracker.get_instance(client_id)
        if ml_tracker.__client_id:
            google_open_id_connect_token = id_token.fetch_id_token(
                Request(), ml_tracker.__client_id
            )
            os.environ["MLFLOW_TRACKING_TOKEN"] = google_open_id_connect_token
