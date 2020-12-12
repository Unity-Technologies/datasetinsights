import logging
import tempfile
from pathlib import Path

from yacs.config import CfgNode

import datasetinsights.constants as const
from datasetinsights.io import create_downloader
from datasetinsights.io.exceptions import DownloadError, InvalidOverrideKey

logger = logging.getLogger(__name__)
_EQUAL_STR = "="


def prepare_config(path=None, override=None):
    """load and override config from local or remote locations .

    Args:
        path: config location
        override: key-value pairs to override config
    Returns:
        config object of type yacs.config.CfgNode
    """
    config = load_config(path)
    logger.info(f"config loading from {path} completed")
    if override:
        logger.info(f"overriding config params= {override}")
        override_config(override=override, config=config)
    return config


def load_config(path):
    """load config from local or remote locations .

    Args:
        path: config location
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
        try:
            downloader = create_downloader(source_uri=path)
            with tempfile.TemporaryDirectory() as tmp:
                downloader.download(source_uri=path, output=tmp)
                logger.info(f"downloading to directory: {tmp}")
                config_path = Path(path)
                file_path = tmp / config_path.relative_to(config_path.parent)
                return CfgNode.load_cfg(open(file_path, "r"))
        except Exception:
            logger.exception(f"Downloading from location{path} failed")
            raise DownloadError

    logger.info(f"loaded config from {path}")
    return CfgNode.load_cfg(open(path, "r"))


def override_config(override=None, config=None):
    """override config params from cli .

    Args:
        override: string of key-value pairs
        config: config object of type yacs.config.CfgNode
    Examples:
         override="optimizer.args.lr=0.00005 pretrained=False"
    """

    try:
        logger.info(f" config before overriding {config}")
        tokens = override.split()
        merge_list = []
        for token in tokens:
            merge_list.extend(token.split(_EQUAL_STR))
        logger.info(f" overriding key-values {merge_list}")
        config.merge_from_list(merge_list)
        logger.info(f" overriding completed, config after override{config}")
    except AssertionError:
        logger.exception("requested override key does not exist")
        raise InvalidOverrideKey
    except Exception as e:
        logger.exception("overriding failed")
        raise e
