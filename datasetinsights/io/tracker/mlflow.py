import os

import mlflow
from google.auth.transport.requests import Request
from google.oauth2 import id_token


class MLFlowTracker:
    """ mlflow wrapper class, responsible for setting host, client_id and return
        initialized mlflow.

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
        __client_id: tracking server client id
        __instance: holds singleton instance
        __mlflow: holds initialized mlflow

    """

    __instance = None
    __mlflow = None
    __client_id = None

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
                self.__client_id = client_id
                MLFlowTracker.refresh_token(client_id)
            mlflow.set_tracking_uri(host_id)
            if exp_name:
                mlflow.set_experiment(experiment_name=exp_name)
            self.__mlflow = mlflow
            MLFlowTracker.__instance = self

    def __getattr__(self, name):
        """
        if you call ``start_run`` on the instance of this class then
        __getattr__ finds ``start_run`` and passes it to self.__mlflow instance
        """
        if self.__client_id:
            MLFlowTracker.refresh_token(self.__client_id)
        return getattr(self.__mlflow, name)

    @staticmethod
    def get_instance():

        """Static instance access method.
        Returns:
            MLFlowtracker singleton instance.
        """
        return MLFlowTracker.__instance

    @staticmethod
    def refresh_token(client_id):
        """refresh token and set in environment variable.

        Args:
            client_id : MLFlow tracking server client id
        """
        if client_id:
            print("refresh_token")
            google_open_id_connect_token = id_token.fetch_id_token(
                Request(), client_id
            )
            os.environ["MLFLOW_TRACKING_TOKEN"] = google_open_id_connect_token


class DummyMLFlowTracker:
    """A fake mfflow writer that writes nothing to the disk. This writer is
    used to disable mlflow logging.
    """

    __instance = None

    def __init__(self, *args, **kwargs):
        if DummyMLFlowTracker.__instance:
            raise Exception("This class is a singleton!")
        else:
            DummyMLFlowTracker.__instance = self

    @staticmethod
    def get_instance():

        """Static instance access method.
        Returns:
            DummyMLFlowTracker singleton instance.
        """
        return DummyMLFlowTracker.__instance

    def start_run(self, *args, **kwargs):
        return

    def end_run(self, *args, **kwargs):
        return

    def log_params(self, *args, **kwargs):
        return

    def log_param(self, *args, **kwargs):
        return

    def log_metric(self, *args, **kwargs):
        return

    def log_metrics(self, *args, **kwargs):
        return

    def active_run(self, *args, **kwargs):
        return

    def log_artifact(self, *args, **kwargs):
        return

    def log_artifacts(self, *args, **kwargs):
        return
