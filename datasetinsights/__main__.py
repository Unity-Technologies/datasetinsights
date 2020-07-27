import click
import argparse
import logging
import sys

import torch
from tensorboardX import SummaryWriter
from yacs.config import CfgNode as CN

import datasetinsights.constants as const

from .configs import system
from .data.datasets import Dataset
from .estimators import Estimator
from .storage.checkpoint import create_checkpointer
from .storage.kfp_output import KubeflowPipelineWriter
from .torch_distributed import get_world_size, init_distributed_mode, is_master

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(levelname)s | %(asctime)s | %(name)s | %(threadName)s | "
        "%(message)s"
    ),
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@click.group()
@click.option(
    "--local_rank",
    help="the local rank of the subprocess excecuting the code. "
    "If sequential, this should be 0."
    "Otherwise torch.distributed.launch will specify this arg.",
    type=int,
)
@click.option(
    "-v",
    "--verbose",
    default=system.verbose,
    help="turn on verbose mode to enable debug logging",
)
@click.option(
    "--dryrun",
    default=system.dryrun,
    help="dry run with only a small subset of the dataset",
)
@click.option(
    "--logdir",
    type=str,
    default=system.logdir,
    help="path where to save training logs",
)
@click.option(
    "--data-root",
    type=str,
    default=system.data_root,
    help="root directory of all datasets",
)
@click.option(
    "--val-interval",
    type=int,
    default=system.val_interval,
    help="control how many training epochs between each validation run",
)
@click.argument(
    "-c", "--config", type=str, help="config file for this model",
)
@click.option(
    "-j",
    "--workers",
    type=int,
    default=system.workers,
    help="number of data loading workers",
)
def main():
    pass


@main.command()
@click.option(
    "--auth-token",
    type=str,
    default=system.auth_token,
    help=(
        "authorization token to fetch USim manifest file that speficed "
        "signed URLs for the simulation dataset files."
    ),
)
def download():
    pass


@main.command()
@click.option(
    "--gpus-per-node",
    default=get_world_size(),
    help="number of gpus per node to use for training/evaluating",
)
@click.option(
    "--no-cuda", default=system.no_cuda, help="force to turn off cuda training",
)
@click.option(
    "--metricsdir",
    type=str,
    default=system.metricsdir,
    help="path where to save metrics",
)
@click.option(
    "--metricsfilename",
    type=str,
    default=system.metricsfilename,
    help="path where to save metrics",
)
def train():
    pass


@main.command()
@click.option(
    "--gpus-per-node",
    default=get_world_size(),
    help="number of gpus per node to use for training/evaluating",
)
@click.option(
    "--no-cuda", default=system.no_cuda, help="force to turn off cuda training",
)
@click.option(
    "--metricsdir",
    type=str,
    default=system.metricsdir,
    help="path where to save metrics",
)
@click.option(
    "--metricsfilename",
    type=str,
    default=system.metricsfilename,
    help="path where to save metrics",
)
def evaluate():
    pass


main()
