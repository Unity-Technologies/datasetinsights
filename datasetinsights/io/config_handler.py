import logging
import re
import tempfile
from pathlib import Path

from yacs.config import CfgNode

import datasetinsights.constants as const
from datasetinsights.io import create_downloader

logger = logging.getLogger(__name__)
_REMOTE_PATTERN = r"://"

""" This module handles YAML config related operations such as
    locating config from local or remote locations.
"""


def load_config(path):
    """ Load config file from local or remote locations.

    Args:
        path (str): This is the file-uri that indicates where
                          the YAML config should be loaded from.
        Examples:
            >>> path = "gs://thea-dev/config.yaml"
            >>> path = "https://thea-dev/config.yaml"
            >>> path = "http://thea-dev/config.yaml"
            >>> path = "file:///root/config.yaml" # absolute path
            >>> path = "/root/config.yaml" # absolute path
            >>> path = "datasetinsights/config.yaml" # relative path

    Returns:
        config object of type yacs.config.CfgNode
    """
    logger.info(f"loading config from {path}")
    if path.startswith(const.LOCAL_FILE_BASE_STR):
        path = path[len(const.LOCAL_FILE_BASE_STR) :]
    if re.search(_REMOTE_PATTERN, path):
        downloader = create_downloader(source_uri=path)
        with tempfile.TemporaryDirectory() as tmp:
            downloader.download(source_uri=path, output=tmp)
            logger.debug(f"downloading to directory: {tmp}")
            config_path = Path(path)
            file_path = tmp / config_path.relative_to(config_path.parent)
            return CfgNode.load_cfg(open(file_path, "r"))

    return CfgNode.load_cfg(open(path, "r"))
