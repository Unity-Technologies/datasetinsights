import argparse
import logging
import sys

import torch
from tensorboardX import SummaryWriter
from yacs.config import CfgNode as CN

import datasetinsights.constants as const
from datasetinsights.datasets import Dataset

from .configs import system
from .estimators import Estimator
from .storage.checkpoint import EstimatorCheckpoint
from .storage.kfp_output import KubeflowPipelineWriter
from .torch_distributed import init_distributed_mode, is_master

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
    command_parser = argparse.ArgumentParser(
        description="Datasetinsights Modeling Interface"
    )
    command_parser.add_argument(
        "command",
        help="sub-command to run",
        choices=("train", "evaluate", "download-train", "download-evaluate"),
    )

    parser = argparse.ArgumentParser(
        description="Datasetinsights Modeling Interface"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=system.verbose,
        help="turn on verbose mode to enable debug logging",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        default=system.no_cuda,
        help="force to turn off cuda training",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default=system.logdir,
        help="path where to save training logs",
    )
    parser.add_argument(
        "--metricsdir",
        type=str,
        default=system.metricsdir,
        help="path where to save metrics",
    )
    parser.add_argument(
        "--metricsfilename",
        type=str,
        default=system.metricsfilename,
        help="path where to save metrics",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=system.data_root,
        help="root directory of all datasets",
    )

    parser.add_argument(  # This has to be moved to estimator config
        "--val-interval",
        type=int,
        default=system.val_interval,
        help="control how many training epochs between each validation run",
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        metavar="N",
        default=system.workers,
        help="number of data loading workers",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="config file for this model",
    )
    parser.add_argument(  # stretch goal to support override
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="extra options to override config file",
    )
    cmd = command_parser.parse_args(sys.argv[2:3]).command
    cmd_args = parser.parse_args([sys.argv[1]] + sys.argv[3:])
    logger.debug("Parsed CLI args: %s", cmd_args)

    return cmd, cmd_args


def parse_config(args):
    cfg = CN.load_cfg(open(args.config, "r"))
    cfg.merge_from_list(args.opts)

    return cfg


def run(command, cfg):
    if cfg.system.verbose:
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

    logger.info("Run command: %s with config: %s\n", command, cfg)

    if torch.cuda.is_available() and not cfg.system.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logdir = cfg.system.logdir
    if logdir == const.NULL_STRING:
        # Use logdir=None to force using SummaryWriter default logdir,
        # which points to ./runs/<model>_<timestamp>
        logdir = None

    # todo this makes it so that we lose the tensorboard writer of non-master
    # processes which could make debugging harder
    writer = SummaryWriter(logdir, write_to_disk=is_master())
    kfp_writer = KubeflowPipelineWriter(
        filename=cfg.system.metricsfilename, filepath=cfg.system.metricsdir
    )
    checkpointer = EstimatorCheckpoint(
        estimator_name=cfg.estimator,
        log_dir=writer.logdir,
        distributed=cfg.system.distributed,
    )
    estimator = Estimator.create(
        cfg.estimator,
        config=cfg,
        writer=writer,
        kfp_writer=kfp_writer,
        device=device,
        checkpointer=checkpointer,
        gpu=args.gpu,
        rank=args.rank,
    )

    if command == "train":
        estimator.train()
    elif command == "evaluate":
        estimator.evaluate()
    elif command == "download-train":
        # TODO (YC)
        # We should remove reference to auth-token in various places to
        # enable download synthetic dataset. Usim is working on a solution
        # that will enable customers to sprcify cloud storage path
        # to store simulations. In the future, we should simply rely
        # on gcs service accounts to access simulation data for a given
        # run execution id.
        Dataset.create(
            cfg.train.dataset.name,
            data_root=cfg.system.data_root,
            auth_token=cfg.system.auth_token,  # XXX(YC) This should be removed
            **cfg.train.dataset.args,
        )
        Dataset.create(
            cfg.val.dataset.name,
            data_root=cfg.system.data_root,
            auth_token=cfg.system.auth_token,  # XXX(YC) This should be removed
            **cfg.val.dataset.args,
        )
    elif command == "download-evaluate":
        Dataset.create(
            cfg.test.dataset.name,
            data_root=cfg.system.data_root,
            auth_token=cfg.system.auth_token,  # XXX(YC) This should be removed
            **cfg.test.dataset.args,
        )

    writer.close()
    kfp_writer.write_metric()


if __name__ == "__main__":
    command, args = parse_args()
    init_distributed_mode(args)
    cfg = parse_config(args)
    run(command, cfg)
