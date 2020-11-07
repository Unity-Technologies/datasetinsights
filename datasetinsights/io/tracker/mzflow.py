import os

import mlflow
from google.auth.transport.requests import Request
from google.oauth2 import id_token


class MLFlowTracker:
    def __init__(self):
        pass

    @staticmethod
    def get_tracker(config):

        client_id = config.get("mlflow").get("server_client_id")
        host_id = config.get("mlflow").get("server_host_id")
        if host_id:
            if client_id:
                google_open_id_connect_token = id_token.fetch_id_token(
                    Request(), client_id
                )
                # Set environment variables
                os.environ[
                    "MLFLOW_TRACKING_TOKEN"
                ] = google_open_id_connect_token
            mlflow.set_tracking_uri(host_id)
            mlflow.set_experiment(
                experiment_name="datasetinsights_kubeflow_run"
            )  #
        config.pop("mlflow", None)
        return mlflow
