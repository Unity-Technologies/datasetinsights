import logging
import os
import tempfile

from yacs.config import CfgNode

from datasetinsights.constants import GCS_BASE_STR
from datasetinsights.io import create_downloader

logger = logging.getLogger(__name__)


class ConfigHandler:
    """ Handles all the config related tasks such as
     loading config from local or remote locations.
    """

    @staticmethod
    def load_config(config):
        """load config from local or remote location .

        Args:
            config: config location
        Returns:
            loaded config object of type dictionary
        """
        if config.startswith(GCS_BASE_STR):
            downloader = create_downloader(source_uri=config)
            with tempfile.TemporaryDirectory() as tmp:
                downloader.download(source_uri=config, output=tmp)
                logger.info(f"downloading to directory: {tmp}")
                file_path = os.path.join(tmp, config.split("/")[-1])
                logger.info(f"loaded config from {config}")
                return CfgNode.load_cfg(open(file_path, "r"))
        logger.info(f"loaded config from {config}")
        return CfgNode.load_cfg(open(config, "r"))
