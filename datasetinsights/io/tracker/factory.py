import logging
import threading

from datasetinsights.io.exceptions import InvalidTrackerError
from datasetinsights.io.tracker.mlflow import MLFlowTracker

logger = logging.getLogger(__name__)


class TrackerFactory:
    """Factory: responsible for creating and holding singleton instance
        of tracker classes"""

    TRACKER = "tracker"
    HOST_ID = "host"
    MLFLOW_TRACKER = "mlflow"
    __singleton_lock = threading.Lock()
    __tracker_instance = None
    RUN_FAILED = "FAILED"

    @staticmethod
    def create(config=None, tracker_type=None):
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
                if mlflow_config.get(TrackerFactory.HOST_ID, None):
                    try:
                        mlf_tracker = TrackerFactory._mlflow_tracker_instance(
                            mlflow_config
                        ).get_mlflow()
                        logger.info("initializing mlflow_tracker")
                        return mlf_tracker
                    except Exception as e:
                        logger.warning(
                            "failed mlflow initialization, "
                            "starting null_tracker",
                            e,
                        )

            logger.info("initializing null_tracker")
            return TrackerFactory._null_tracker()
        else:
            logger.exception(f"Unknown tracker {tracker_type}!")
            raise InvalidTrackerError

    @staticmethod
    def _mlflow_tracker_instance(mlflow_config):

        """Static instance access method.

        Args:
            host_id: MlTracker server host
            client_id: MLFlow tracking server client id
            exp_name: name of the experiment
        Returns:
            tracker singleton instance.
        """
        if not TrackerFactory.__tracker_instance:
            with TrackerFactory.__singleton_lock:
                if not TrackerFactory.__tracker_instance:
                    TrackerFactory.__tracker_instance = MLFlowTracker(
                        mlflow_config
                    )
        logger.info("getting tracker instance")
        return TrackerFactory.__tracker_instance

    @staticmethod
    def _null_tracker():
        """private getter method to get singleton instance.

        Returns:
            NullTracker singleton instance.
        """
        if not TrackerFactory.__tracker_instance:
            with TrackerFactory.__singleton_lock:
                if not TrackerFactory.__tracker_instance:
                    TrackerFactory.__tracker_instance = NullTracker()
        logger.info("getting null tracker")
        return TrackerFactory.__tracker_instance


class NullTracker:
    """A null tracker that writes nothing. This tracker is
    used to disable tracking.
    """

    def _stdout_handler(self, *args, **kwargs):
        """method to handle all calls on tracker.
        """
        return

    def __getattr__(self, name):
        """gets called at every call on the instance of this class.
        """
        logger.info(f"null_tracker handling {name}")
        return getattr(self, "_stdout_handler")
