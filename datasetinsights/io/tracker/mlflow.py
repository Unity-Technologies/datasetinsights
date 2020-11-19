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
        __instance: holds singleton instance
        __mlflow: holds initialized mlflow
        __client_id: tracking server client id
        __oauth_token: holds oauth token returns by IAP

    """

    __instance = None
    __mlflow = None
    __client_id = None
    __oauth_token = None

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
                self.refresh_token()
            mlflow.set_tracking_uri(host_id)
            if exp_name:
                mlflow.set_experiment(experiment_name=exp_name)
            self.__mlflow = mlflow
            MLFlowTracker.__instance = self

    def __getattr__(self, name):
        """
        if you call any method on this class instance then
        __getattr__ finds that method in __mlflow and calls __mlflow.instance
        Args:
            name: method name which you want to call on this class instance
        Returns:
            return what name method returns
        """
        if self.__client_id:
            try:
                id_token.verify_oauth2_token(
                    self.__oauth_token, Request(), self.__client_id
                )
            except ValueError:
                self.refresh_token()
        return getattr(self.__mlflow, name)

    @staticmethod
    def get_instance():

        """Static instance access method.
        Returns:
            MLFlowtracker singleton instance.
        """
        return MLFlowTracker.__instance

    def refresh_token(self):
        """refresh token and set in environment variable
        """
        if self.__client_id:
            google_open_id_connect_token = id_token.fetch_id_token(
                Request(), self.__client_id
            )
            os.environ["MLFLOW_TRACKING_TOKEN"] = google_open_id_connect_token
            self.__oauth_token = google_open_id_connect_token


class DummyMLFlowTracker:
    """A fake mlflow tracker that writes nothing. This tracker is
    used to disable mlflow tracking.
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
