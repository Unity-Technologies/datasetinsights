import argparse
import logging
import os
import zipfile

import datasetinsights.constants as const
from datasetinsights.data.download import download_file

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
        description="Download Dataset from Public URL or GCS Bucket"
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
        "--dataset-uri", default="None", help="URL of synthdet dataset.",
    )

    args = parser.parse_args()
    return args


def unzip_file(filepath, destination):
    """Unzips a zip file to the destination and delete the zip file.

    Args:
        filepath (str): File path of the zip file.
        destination (str): Path where to unzip contents of zipped file.
    """
    try:
        with zipfile.ZipFile(filepath) as file:
            logger.info("Unzipping file.")
            file.extractall(destination)
        os.remove(filepath)
    except zipfile.BadZipFile:
        logger.error("Zip file is corrupted.")


def run(args):

    if args.verbose:
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

    logger.info("Run command with args: %s\n", args)
    data_root = args.data_root
    dataset_uri = args.dataset_uri

    if dataset_uri != "None":

        if dataset_uri.startswith(
            (const.HTTP_URL_BASE_STR, const.HTTPS_URL_BASE_STR)
        ):
            logger.info("Downloading dataset to data root.")
            dataset_path = download_file(
                dataset_uri, os.path.join(data_root, "dataset.zip")
            )
            unzip_file(filepath=dataset_path, destination=data_root)

        else:
            raise ValueError(
                f"Given URL: {dataset_uri}, is either invalid or not supported."
                f"Currently supported path is HTTP url (http:// or https://) "
                f"path"
            )
    else:
        logger.info(
            f"No dataset_uri is provided. Assuming the data root directory"
            f" {data_root} already contains synthetic dataset."
        )


if __name__ == "__main__":
    args = parse_args()
    run(args)
