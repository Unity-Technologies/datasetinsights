""" Download Public GroceriesReal Dataset from GCS
"""
import argparse
import logging
import os

import datasetinsights.constants as const
from datasetinsights.data.datasets import GroceriesReal
from datasetinsights.data.download import compare_checksums, download_file
from datasetinsights.data.exceptions import DownloadError

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
    parser = argparse.ArgumentParser(
        description="Download GroceriesReal dataset from a public GCS path"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="turn on verbose mode to enable debug logging",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=const.DEFAULT_DATA_ROOT,
        help="root directory of datasets",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=const.DEFAULT_DATA_ROOT,
        help="dataset name, e.g. GroceriesReal, Synthetic",
        required=True,
    )
    parser.add_argument(
        "--version", type=str, default="v3", help="dataset version, e.g. v3",
    )

    args = parser.parse_args()

    return args


def run(args):
    if args.verbose:
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

    logger.info("Run command with args: %s\n", args)
    data_root = args.data_root

    file_cloud_path = GroceriesReal.GROCERIES_REAL_DATASET_TABLES[
        args.version
    ].http_path
    file_dest_path = os.path.join(data_root, LOCAL_PATH, f"{args.version}.zip")
    download_file(source_uri=file_cloud_path, dest_path=file_dest_path)
    checksum_path = GroceriesReal.GROCERIES_REAL_DATASET_TABLES[
        args.version
    ].checksum
    checksum_dest_path = os.path.join(
        data_root, LOCAL_PATH, f"{args.version}.txt"
    )
    download_file(source_uri=checksum_path, dest_path=checksum_dest_path)
    if not compare_checksums(
        file_path=file_dest_path, checksum_path=checksum_dest_path
    ):
        raise DownloadError(
            f"Invalid hash value."
            f"There are some errors during the dataset download. "
            f"Please download it again."
        )


if __name__ == "__main__":
    args = parse_args()
    run(args)
