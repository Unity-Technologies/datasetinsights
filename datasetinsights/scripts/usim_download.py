""" Download Synthetic Dataset from Unity Simulation Platform

[Unity Simulation](https://unity.com/products/simulation) provides a powerful
platform for running simulations at large scale. This script provides basic
functionality that allow users to download generated synthetic dataset.
"""
import argparse
import logging
import os
from pathlib import Path

import datasetinsights.constants as const
from datasetinsights.data.simulation.download import (
    Downloader,
    download_manifest,
)

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
        description="Download Dataset from Unity Simulation Platform"
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
        "--auth-token",
        type=str,
        help=(
            "authorization token to fetch USim manifest file that specified "
            "signed URLs for the simulation dataset files."
        ),
    )
    parser.add_argument(
        "--project-id",
        type=str,
        help="Unity Project ID",
        default=const.DEFAULT_PROJECT_ID,
    )
    parser.add_argument(
        "--run-execution-id",
        type=str,
        help=("USim run-execution-id"),
        required=True,
    )
    parser.add_argument(
        "--include-binary",
        action="store_true",
        default=False,
        help="include binary files such as captured images.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        default=False,
        help="toggle to force re-download by ignoring cache files",
    )

    args = parser.parse_args()

    return args


def run(args):
    if args.verbose:
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

    logger.info("Run command with args: %s\n", args)
    data_root = args.data_root

    auth_token = args.auth_token
    use_cache = not args.no_cache
    run_execution_id = args.run_execution_id

    manifest_file = os.path.join(
        data_root, const.SYNTHETIC_SUBFOLDER, f"{run_execution_id}.csv"
    )
    if auth_token:
        manifest_file = download_manifest(
            run_execution_id,
            manifest_file,
            auth_token,
            project_id=args.project_id,
            use_cache=use_cache,
        )
    else:
        logger.info(
            f"No auth token is provided. Assuming you already have "
            f"a manifest file located in {manifest_file}"
        )

    subfolder = Path(manifest_file).stem
    root = os.path.join(data_root, const.SYNTHETIC_SUBFOLDER, subfolder)

    dl_worker = Downloader(manifest_file, root, use_cache=use_cache)
    dl_worker.download_references()
    dl_worker.download_metrics()
    dl_worker.download_captures()
    if args.include_binary:
        dl_worker.download_binary_files()


if __name__ == "__main__":
    args = parse_args()
    run(args)
