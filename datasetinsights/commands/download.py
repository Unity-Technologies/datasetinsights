import logging
import re

import click

import datasetinsights.constants as const
from datasetinsights.data.datasets import Downloader

logger = logging.getLogger(__name__)


class SourceURI(click.ParamType):
    """Source URI Parameter.

    Args:
        click ([type]): [description]

    Raises:
        click.BadParameter: [description]

    Returns:
        [type]: [description]
    """

    name = "source_uri"
    PREFIX_PATTERN = r"^gs://|^http(s)?://|^usim://"

    def convert(self, value, param, ctx):
        """ Validate source URI and Converts the value.
        """
        match = re.search(self.PREFIX_PATTERN, value)
        if not match:
            message = (
                f"The source uri {value} is not supported. "
                f"Pattern: {self.PREFIX_PATTERN}"
            )
            self.fail(message, param, ctx)

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
    type=click.Path(exists=True, file_okay=False, writable=True),
    default=const.DEFAULT_DATA_ROOT,
    help="Root directory on localhost where datasets should be downloaded.",
)
@click.option(
    "-s",
    "--source-uri",
    type=SourceURI(),
    default=None,
    help=(
        "URI of where this data should be downloaded. "
        "If not supplied, default path from the dataset registry will be used. "
        f"Supported source uri patterns {SourceURI.PREFIX_PATTERN}"
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
    downloader = Downloader.create(
        name,
        data_root=data_root,
        source_uri=source_uri,
        project_id=project_id,
        run_execution_id=run_execution_id,
        auth_token=auth_token,
        version=dataset_version,
    )
    downloader.download(include_binary=include_binary)
