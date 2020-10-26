import json
import logging
import os

from datasetinsights.constants import DEFAULT_TENSORBOARD_LOG_DIR

logger = logging.getLogger(__name__)


class KubeflowPipelineWriter(object):
    """
    Serializes metrics dictionary genereated during model training/evaluation to
    JSON and store in a file.

    Args:
        filename (str): Name of the file to which the writer will save metrics
        filepath (str): Path where the file will be stored

    Attributes:
        filename (str): Name of the file to which the writer will save metrics
        filepath (str): Path where the file will be stored
        data_dict (dict): A dictionary to save metrics name and value pairs
        data: Dictionary to be JSON serialized
    """

    def __init__(
        self,
        tb_log_dir=DEFAULT_TENSORBOARD_LOG_DIR,
        filename="mlpipeline-metrics.json",
        filepath="/",
    ):
        """
        Creates KubeflowPipelineWriter that will write out metrics to the output
        file
        """

        self.filename = filename
        self.filepath = filepath
        self.data_dict = {}
        self.data = {"metrics": []}
        self.tb_log_dir = tb_log_dir

    def add_metric(self, name, val):
        """
        Adds metric to the data dictionary of the writer

        Note: Using same name key will overwrite the previous value as the
        current strategy is to save only the metrics generated in last epoch

        Args:
            name (str): Name of the metric
            val (float): Value of the metric
        """

        logger.debug("Metric {0} with value: {1} added".format(name, val))
        self.data_dict[name] = val

    def write_metric(self):
        """
        Saves all the metrics added previously to a file in the format required
        by kubeflow
        """
        if not self.data_dict:
            logger.warning("No metrics generated to be saved.")
            return

        for key, val in self.data_dict.items():
            self.data["metrics"].append(
                {"name": key, "numberValue": val, "format": "RAW"}
            )
        if not os.path.exists(self.filepath):
            os.makedirs(self.filepath)
        with open(os.path.join(self.filepath, self.filename), "w") as f:
            json.dump(self.data, f)

        logger.debug(
            f"Metrics file {self.filename} saved at path:" f" {self.filepath}"
        )

    def create_tb_visualization_json(self):
        try:
            metadata = {
                "outputs": [{"type": "tensorboard", "source": self.tb_log_dir}]
            }
            with open("/mlpipeline-ui-metadata.json", "w") as f:
                json.dump(metadata, f)

        # when we don't have write permission
        except IOError:
            logger.info("Can not create Tensorboard Visualization JSON file.")
