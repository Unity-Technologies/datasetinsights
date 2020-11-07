from datasetinsights.io.tracker.mzflow import MLFlowTracker


class TrackerFactory:
    def __init__(self):
        pass

    @staticmethod
    def create(config, tracker_type):

        return MLFlowTracker.get_tracker(config)
