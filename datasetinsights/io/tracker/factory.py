from datasetinsights.constants import TIMESTAMP_SUFFIX
from datasetinsights.io.tracker.mlflow import DummyMLFlowTracker, MLFlowTracker


class TrackerFactory:
    """Factory: responsible for creating and holding singleton instance
        of tracker classes"""

    TRACKER = "tracker"
    CLIENT_ID = "client_id"
    HOST_ID = "host"
    EXP_NAME = "experiment"
    RUN_NAME = "run"
    MLFLOW_TRACKER = "mlflow"
    DEFAULT_RUN_NAME = "run_" + TIMESTAMP_SUFFIX

    @staticmethod
    def create(config, tracker_type):
        """create tracker class object.

        Args:
            config : config object, holds server details
            tracker_type : type of tracker

        Returns:
            tracker instance.
        """
        if TrackerFactory.MLFLOW_TRACKER == tracker_type:

            tracker = config.get(TrackerFactory.TRACKER, None)
            if tracker and tracker.get(TrackerFactory.MLFLOW_TRACKER, None):
                mlflow_config = tracker.get(TrackerFactory.MLFLOW_TRACKER)
                host_id = mlflow_config.get(TrackerFactory.HOST_ID, None)
                if host_id:
                    client_id = mlflow_config.get(
                        TrackerFactory.CLIENT_ID, None
                    )
                    exp_name = mlflow_config.get(TrackerFactory.EXP_NAME, None)
                    TrackerFactory.update_run_name(mlflow_config)
                    mlflow_tracker = TrackerFactory.get_tracker_instance(
                        host_id=host_id, client_id=client_id, exp_name=exp_name
                    )
                    mlflow_tracker.start_run(
                        run_name=TrackerFactory.DEFAULT_RUN_NAME
                    )
                    return mlflow_tracker
            return TrackerFactory.get_tracker_instance()
        else:
            raise NotImplementedError(f"Unknown tracker {tracker_type}!")

    @staticmethod
    def get_tracker_instance(host_id=None, client_id=None, exp_name=None):

        """Static instance access method.

        Args:
            host_id: MlTracker server host
            client_id: MLFlow tracking server client id
            exp_name: name of the experiment
        Returns:
            tracker singleton instance.
        """
        if host_id:
            if not MLFlowTracker.get_instance():
                MLFlowTracker(
                    host_id=host_id, client_id=client_id, exp_name=exp_name
                )
            return MLFlowTracker.get_instance()

        if not DummyMLFlowTracker.get_instance():
            DummyMLFlowTracker()
        return DummyMLFlowTracker.get_instance()

    @staticmethod
    def update_run_name(mlflow_config):

        """Static method to update run name from config.

        Args:
            mlflow_config : mlflow_config dictionary
        """
        run_name = mlflow_config.get(TrackerFactory.RUN_NAME, None)
        if run_name:
            TrackerFactory.DEFAULT_RUN_NAME = run_name
