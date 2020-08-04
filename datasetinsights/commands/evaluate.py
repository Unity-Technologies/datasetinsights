import logging

import click

import datasetinsights.constants as const

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
    "-d",
    "--data-root",
    type=click.STRING,
    default=const.DEFAULT_DATA_ROOT,
    help="Root directory on localhost where datasets are located.",
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
    type=click.STRING,
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
    data_root,
    tb_log_dir,
    workers,
    kfp_metrics_dir,
    kfp_metrics_filename,
    no_cuda,
):
    ctx = click.get_current_context()
    logger.debug(f"Called evaluate command with parameters: {ctx.params}")
    logger.debug(f"Override estimator config with args: {ctx.args}")
    # TODO: Call evaluate command here.
