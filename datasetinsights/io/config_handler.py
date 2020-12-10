import logging
import tempfile
from pathlib import Path

from yacs.config import CfgNode

import datasetinsights.constants as const
from datasetinsights.io import create_downloader

logger = logging.getLogger(__name__)


def handle_config(path=None, access_token=None):
    """load and override config from local or remote location .

    Args:
        path: config location
        access_token: authenticate remote location
    Examples:
         path="gs://thea-dev/../config.yaml"
         path="https://thea-dev/../config.yaml"
         path="http://thea-dev/../config.yaml"
         path="/root/../config.yaml"
    Returns:
        config object of type yacs.config.CfgNode
    """
    if path.startswith(
        (const.GCS_BASE_STR, const.HTTP_URL_BASE_STR, const.HTTPS_URL_BASE_STR,)
    ):
        downloader = create_downloader(
            source_uri=path, access_token=access_token
        )
        with tempfile.TemporaryDirectory() as tmp:
            downloader.download(source_uri=path, output=tmp)
            logger.info(f"downloading to directory: {tmp}")
            config_path = Path(path)
            file_path = tmp / config_path.relative_to(config_path.parent)
            logger.info(f"loaded config from {path}")
            return CfgNode.load_cfg(open(file_path, "r"))
    logger.info(f"loaded config from {path}")
    return CfgNode.load_cfg(open(path, "r"))
