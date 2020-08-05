""" Download dataset.
"""
import argparse
import logging

import datasetinsights.constants as const
from datasetinsights.data.datasets import Dataset, SynDetection2D

LOCAL_PATH = "groceries"

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(levelname)s | %(asctime)s | %(name)s | %(threadName)s | "
        "%(message)s"
    ),
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Download dataset")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="turn on verbose mode to enable debug logging",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=const.DEFAULT_PUBLIC_DATASET,
        help="dataset name, e.g. GroceriesReal, Synthetic",
        required=True,
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=const.DEFAULT_DATA_ROOT,
        help="root directory of datasets",
    )
    parser.add_argument(
        "--source-uri", default="None", help="URL of synthdet dataset.",
    )

    args = parser.parse_args()

    return args


def download(name, source_uri, data_root, version):
    # TODO this method should be refactored once we have clean download
    # CLI interface.
    if name.lower() == "Synthetic".lower():
        SynDetection2D.download(source_uri, data_root)
    else:
        dataset = Dataset.find(name)
        dataset.download(data_root, version)


def run(args):

    if args.verbose:

        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
    logger.info("Run command with args: %s\n", args)

    download(args.name, args.source_uri, args.data_root, version="v3")


if __name__ == "__main__":
    args = parse_args()
    # TODO this method should be refactored once we have clean download
    run(args)
