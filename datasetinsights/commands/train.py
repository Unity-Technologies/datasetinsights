import logging

import click
from yacs.config import CfgNode as CN

import datasetinsights.constants as const
from datasetinsights.estimators.base import create_estimator

logger = logging.getLogger(__name__)


@click.command(
    help="Start model training (and optionally validation) tasks.",
    context_settings=const.CONTEXT_SETTINGS,
)
@click.option(
    "-c",
    "--config",
    type=click.STRING,
    required=True,
    help="Path to the config estimator yaml file.",
)
@click.option(
    "-t",
    "--train-data",
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help="Directory on localhost where train dataset is located.",
)
@click.option(
    "-e",
    "--val-data",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help="Directory on localhost where validation dataset is located.",
)
@click.option(
    "-p",
    "--checkpoint-file",
    type=click.STRING,
    default=None,
    help=(
        "URI to a checkpoint file. If specified, model will load from "
        "this checkpoint and resume training."
    ),
)
@click.option(
    "-l",
    "--tb-log-dir",
    type=click.STRING,
    default=const.DEFAULT_TENSORBOARD_LOG_DIR,
    help=(
        "Path to the directory where tensorboard events should be stored. "
        "This Path can be GCS URI (e.g. gs://<bucket>/runs) or full path "
        "to a local directory."
    ),
)
@click.option(
    "-p",
    "--checkpoint-dir",
    type=click.STRING,
    default=const.DEFAULT_CHECKPOINT_DIR,
    help=(
        "Path to the directory where model checkpoint files should be stored. "
        "This Path can be GCS URI (e.g. gs://<bucket>/checkpoints) or "
        "full path to a local directory."
    ),
)
@click.option(
    "-w",
    "--workers",
    type=click.INT,
    default=0,
    help=(
        "Number of multiprocessing workers for loading datasets. "
        "Set this argument to 0 will disable multiprocessing which is "
        "recommended when running inside a docker container."
    ),
)
@click.option(
    "--no-cuda",
    is_flag=True,
    default=False,
    help=(
        "Force to disable CUDA. If CUDA is available and this flag is False, "
        "model will be trained using CUDA."
    ),
)
@click.option(
    "--no-val",
    is_flag=True,
    default=False,
    help="Force to disable validations.",
)
def cli(
    config,
    train_data,
    val_data,
    checkpoint_file,
    tb_log_dir,
    checkpoint_dir,
    workers,
    no_cuda,
    no_val,
):
    ctx = click.get_current_context()
    logger.debug(f"Called train command with parameters: {ctx.params}")
    logger.debug(f"Override estimator config with args: {ctx.args}")

    config = CN.load_cfg(open(config, "r"))

    estimator = create_estimator(
        name=config.estimator,
        config=config,
        checkpoint_file=checkpoint_file,
        tb_log_dir=tb_log_dir,
        checkpoint_dir=checkpoint_dir,
        no_cuda=no_cuda,
        no_val=no_val,
        workers=workers,
    )

    estimator.train(train_data=train_data, val_data=val_data)
