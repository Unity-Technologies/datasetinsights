""" Ad-Hoc script to run memory profiling for SynthDetection2D

pip install memory_profiler
"""
import logging
import argparse

from tqdm import tqdm
from memory_profiler import profile

from datasetinsights.data.datasets.synthetic import SynDetection2D
import datasetinsights.constants as const


logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(levelname)s | %(asctime)s | %(name)s | %(threadName)s | "
        "%(message)s"
    ),
    datefmt="%Y-%m-%d %H:%M:%S",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Experiments using Kubeflow Pipeline for profiling"
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default=const.DEFAULT_DATA_ROOT,
        help="root directory of datasets",
    )
    parser.add_argument(
        "--run-execution-id",
        help="USim run execution id.",
        default="wjZR6Zj",
        type=str,
    )
    args = parser.parse_args()

    return args


@profile
def load_data(args):
    definition_id = "c31620e3-55ff-4af6-ae86-884aa0daa9b2"
    data = SynDetection2D(
        data_root=args.data_root,
        run_execution_id=args.run_execution_id,
        def_id=definition_id,
    )

    for i in tqdm(range(len(data))):
        data[i]


if __name__ == "__main__":
    args = parse_args()
    load_data(args)
