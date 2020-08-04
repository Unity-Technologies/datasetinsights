import logging

import click

import datasetinsights.constants as const


logger = logging.getLogger(__name__)


def validate_source_uri(ctx, param, value):
    # TODO: validate "value" and raise click.BadParameter exceptions
    # the input value is not valid.
    # https://click.palletsprojects.com/en/7.x/options/#callbacks-for-validation
    return value


@click.command(
    help="Download datasets to localhost from known locations.",
    context_settings=const.CONTEXT_SETTINGS,
)
@click.option(
    "-n",
    "--name",
    type=click.STRING,
    required=True,
    help="The dataset registry name.",
)
@click.option(
    "-d",
    "--data-root",
    type=click.STRING,
    default=const.DEFAULT_DATA_ROOT,
    help="Root directory on localhost where datasets should be downloaded.",
)
@click.option(
    "-s",
    "--source-uri",
    type=click.STRING,
    default=None,
    callback=validate_source_uri,
    help=(
        "URI of where this data should be downloaded. "
        "If not supplied, default path from the dataset registry will be used."
    ),
)
@click.option(
    "-b",
    "--include-binary",
    is_flag=True,
    default=False,
    help=(
        "Whether to download binary files such as images or LIDAR point "
        "clouds. This flag applies to Datasets where metadata "
        "(e.g. annotation json, dataset catalog, ...) can be separated from "
        "binary files."
    ),
)
@click.option(
    "--dataset-version",
    type=click.STRING,
    default=const.DEFAULT_DATASET_VERSION,
    help=(
        "The dataset version. This only applies to some dataset "
        "where different versions are available."
    ),
)
@click.option(
    "--project-id",
    type=click.STRING,
    default=None,
    help=(
        "Unity project-id used for Unity Simulation runs. This will override "
        "synthetic datasets source-uri for Unity Simulation."
    ),
)
@click.option(
    "--run-execution-id",
    type=click.STRING,
    default=None,
    help=(
        "Unity Simulation run-execution-id. This will override synthetic "
        "datasets source-uri for Unity Simulation."
    ),
)
@click.option(
    "--auth-token",
    type=click.STRING,
    default=None,
    help=(
        "Unity Simulation auth token. This will override synthetic "
        "datasets source-uri for Unity Simulation."
    ),
)
def cli(
    name,
    data_root,
    source_uri,
    include_binary,
    dataset_version,
    project_id,
    run_execution_id,
    auth_token,
):
    ctx = click.get_current_context()
    logger.debug(f"Called download command with parameters: {ctx.params}")
    # TODO: Call download dataset command here.
