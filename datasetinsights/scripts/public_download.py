""" Download dataset.
"""
import argparse
import logging

import datasetinsights.constants as const
from datasetinsights.data.datasets import Dataset

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
        "--name",
        type=str,
        default=const.DEFAULT_DATA_ROOT,
        help="dataset name, e.g. GroceriesReal, Synthetic",
        required=True,
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=const.DEFAULT_DATA_ROOT,
        help="root directory of datasets",
    )
    args = parser.parse_args()

    return args


def download(name, data_root, version):
    # TODO this method should be refactored once we have clean download
    # CLI interface.
    dataset = Dataset.find(name)
    dataset.download(data_root, version)


if __name__ == "__main__":
    args = parse_args()
    # TODO this method should be refactored once we have clean download
    # CLI interface.
    download(args.name, args.data_root, version="v3")
