import logging

import click
from yacs.config import CfgNode as CN

import datasetinsights.constants as const
from datasetinsights.estimators.base import create_estimator

logger = logging.getLogger(__name__)


@click.command(
    help="Start model evaluation tasks for a pre-trained model.",
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
    "-p",
    "--checkpoint-file",
    type=click.STRING,
    required=True,
    help="URI to a checkpoint file.",
)
@click.option(
    "-t",
    "--test-data",
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help="Directory on localhost where test dataset is located.",
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
    "--kfp-metrics-dir",
    type=click.Path(file_okay=False, writable=True),
    default=const.DEFAULT_KFP_METRICS_DIR,
    help="Path to the directory where Kubeflow Metrics files are stored.",
)
@click.option(
    "--kfp-metrics-filename",
    type=click.STRING,
    default=const.DEFAULT_KFP_METRICS_FILENAME,
    help="Kubeflow Metrics filename.",
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
def cli(
    config,
    checkpoint_file,
    test_data,
    tb_log_dir,
    workers,
    kfp_metrics_dir,
    kfp_metrics_filename,
    no_cuda,
):
    ctx = click.get_current_context()
    logger.debug(f"Called evaluate command with parameters: {ctx.params}")
    logger.debug(f"Override estimator config with args: {ctx.args}")
    config = CN.load_cfg(open(config, "r"))
    estimator = create_estimator(
        config=config,
        name=config.estimator,
        checkpoint_file=checkpoint_file,
        tb_log_dir=tb_log_dir,
        workers=workers,
        kfp_metrics_dir=kfp_metrics_dir,
        kfp_metrics_filename=kfp_metrics_filename,
        no_cuda=no_cuda,
    )

    estimator.evaluate(test_data=test_data)
