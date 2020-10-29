import json
import logging
import os

from datasetinsights.constants import (
    DEFAULT_KFP_LOG_DIR,
    DEFAULT_KFP_METRICS_FILENAME,
    DEFAULT_KFP_UI_METADATA_FILENAME,
    DEFAULT_TENSORBOARD_LOG_DIR,
)

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
        kfp_log_dir=DEFAULT_KFP_LOG_DIR,
        kfp_metrics_filename=DEFAULT_KFP_METRICS_FILENAME,
        kfp_ui_metadata_filename=DEFAULT_KFP_UI_METADATA_FILENAME,
    ):
        """
        Creates KubeflowPipelineWriter that will write out metrics to the output
        file
        """

        self.kfp_metrics_filename = kfp_metrics_filename
        self.kfp_ui_metadata_filename = kfp_ui_metadata_filename
        self.kfp_log_dir = kfp_log_dir
        self.tb_log_dir = tb_log_dir
        self.data_dict = {}
        self.data = {"metrics": []}
        self.create_tb_visualization_json()

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
        if not os.path.exists(self.kfp_log_dir):
            os.makedirs(self.kfp_log_dir)
        with open(
            os.path.join(self.kfp_log_dir, self.kfp_metrics_filename), "w"
        ) as f:
            json.dump(self.data, f)

        logger.debug(
            f"Metrics file {self.kfp_metrics_filename} saved at path:"
            f" {self.kfp_log_dir}"
        )

    def create_tb_visualization_json(self):

        metadata = {
            "outputs": [{"type": "tensorboard", "source": self.tb_log_dir}]
        }
        if not os.path.exists(self.kfp_log_dir):
            os.makedirs(self.kfp_log_dir)
        with open(
            os.path.join(self.kfp_log_dir, self.kfp_ui_metadata_filename), "w"
        ) as f:
            json.dump(metadata, f)

        logger.debug(
            f"KFP UI Metadata JSON file {self.kfp_ui_metadata_filename} "
            f"saved at path: {self.kfp_log_dir}"
        )
