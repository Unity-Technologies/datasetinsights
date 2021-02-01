import logging
import threading

from mlflow.exceptions import MlflowException

from datasetinsights.io.exceptions import InvalidTrackerError
from datasetinsights.io.tracker.mlflow import MLFlowTracker

logger = logging.getLogger(__name__)


class TrackerFactory:
    """Factory: responsible for creating and holding singleton instance
        of tracker classes"""

    MLFLOW_TRACKER = "mlflow"
    __singleton_lock = threading.Lock()
    __tracker_instance = None
    RUN_FAILED = "FAILED"
    TRACKER = "tracker"

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

            try:
                tracker = config.get(TrackerFactory.TRACKER, None)
                if tracker and tracker.get(TrackerFactory.MLFLOW_TRACKER, None):
                    mlflow_config = tracker.get(TrackerFactory.MLFLOW_TRACKER)

                    mlf_tracker = TrackerFactory._mlflow_tracker_instance(
                        **mlflow_config
                    ).get_mlflow()
                else:
                    mlf_tracker = (
                        TrackerFactory._mlflow_tracker_instance().get_mlflow()
                    )
                logger.info("initializing mlflow_tracker")
                return mlf_tracker
            except ValueError as e:
                logger.warning(
                    "failed mlflow initialization, " "Host is not configured", e
                )
            except MlflowException as e:
                logger.warning("failed mlflow initialization,", e)
            except Exception as e:
                logger.warning(
                    "failed mlflow initialization, " "starting null_tracker", e
                )

            logger.info("initializing null_tracker")
            return TrackerFactory._null_tracker()
        else:
            logger.exception(f"Unknown tracker {tracker_type}!")
            raise InvalidTrackerError

    @staticmethod
    def _mlflow_tracker_instance(**kwargs):

        """Static instance access method.

        Args:
            kwargs : key-value pairs of mlflow parameters
        Returns:
            tracker singleton instance.
        """
        if not TrackerFactory.__tracker_instance:
            with TrackerFactory.__singleton_lock:
                if not TrackerFactory.__tracker_instance:
                    TrackerFactory.__tracker_instance = MLFlowTracker(**kwargs)
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
