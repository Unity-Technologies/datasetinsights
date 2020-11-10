from datasetinsights.constants import MLFLOW_TRACKER
from datasetinsights.io.tracker.mzflow import MLFlowTracker


class TrackerFactory:
    """Factory: responsible for creating instance of tracker class"""

    @staticmethod
    def create(config, tracker_type):
        """create tracker class object.

        Args:
            config : config object, holds server details
            tracker_type : type of tracker
        """
        if MLFLOW_TRACKER == tracker_type:
            return MLFlowTracker.get_tracker(config)
        else:
            raise NotImplementedError(f"Unknown tracker {tracker_type}!")
