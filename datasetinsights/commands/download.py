import logging
import os
import re
from pathlib import Path

import click

import datasetinsights.constants as const
from datasetinsights.datasets.base import DatasetRegistry
from datasetinsights.datasets.simulation.download import (
    Downloader,
    download_manifest,
)

logger = logging.getLogger(__name__)


class Mutex(click.Option):
    def __init__(self, *args, **kwargs):
        self.not_required_if: list = kwargs.pop("not_required_if")

        assert self.not_required_if, "'not_required_if' parameter required"
        kwargs["help"] = (
            kwargs.get("help", "")
            + "Option is mutually exclusive with "
            + ", ".join(self.not_required_if)
            + "."
        ).strip()
        super(Mutex, self).__init__(*args, **kwargs)

    def handle_parse_result(self, ctx, opts, args):
        current_opt: bool = self.name in opts

        for mutex_opt in self.not_required_if:
            if mutex_opt in opts:
                if current_opt:
                    raise click.UsageError(
                        "Illegal usage: '"
                        + str(self.name)
                        + "' is mutually exclusive with "
                        + str(mutex_opt)
                        + "."
                    )
                else:
                    self.prompt = None
        return super(Mutex, self).handle_parse_result(ctx, opts, args)


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
    cls=Mutex,
    not_required_if=["source-uri"],
    type=click.Choice(DatasetRegistry.list_datasets()),
    help="The dataset registry name.",
)
@click.option(
    "-s",
    "--source-uri",
    cls=Mutex,
    type=SourceURI(),
    not_required_if=["name"],
    default=None,
    help=(
        "URI of where this data should be downloaded. "
        "If not supplied, default path from the dataset registry will be used. "
        f"Supported source uri patterns {SourceURI.PREFIX_PATTERN}"
    ),
)
@click.option(
    "-d",
    "--data-root",
    type=click.Path(exists=True, file_okay=False, writable=True),
    default=const.DEFAULT_DATA_ROOT,
    help="Root directory on localhost where datasets should be downloaded.",
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
def cli(
    name, source_uri, data_root, include_binary, dataset_version,
):
    if not name and not source_uri:
        raise click.UsageError("Missing either --name or --source-uri")
    ctx = click.get_current_context()
    logger.debug(f"Called download command with parameters: {ctx.params}")
    if name:
        dataset = DatasetRegistry.find(name)
        dataset.download(
            data_root=data_root,
            version=dataset_version,
            include_binary=include_binary,
        )

    if source_uri.startswith("usim://"):
        usim_download(
            source_uri=source_uri,
            data_root=data_root,
            include_binary=include_binary,
        )

    if source_uri.startswith("https://") or source_uri.startswith("http://"):
        pass

    if source_uri.startswith("gs://"):
        pass


def parse_source_uri_to_usim(source_uri):
    match = re.compile(r"usim://(\w+)@(\w+-\w+-\w+-\w+-\w+)/(\w+)")
    auth_token, project_id, run_execution_id = match.findall(source_uri)[0]
    return auth_token, project_id, run_execution_id


def usim_download(source_uri, data_root, include_binary):
    auth_token, project_id, run_execution_id = parse_source_uri_to_usim(
        source_uri=source_uri
    )

    # use_cache = not args.no_cache
    manifest_file = os.path.join(
        data_root, const.SYNTHETIC_SUBFOLDER, f"{run_execution_id}.csv"
    )
    if auth_token:
        manifest_file = download_manifest(
            run_execution_id, manifest_file, auth_token, project_id=project_id,
        )
    else:
        logger.info(
            f"No auth token is provided. Assuming you already have "
            f"a manifest file located in {manifest_file}"
        )

    subfolder = Path(manifest_file).stem
    root = os.path.join(data_root, const.SYNTHETIC_SUBFOLDER, subfolder)
    dl_worker = Downloader(manifest_file, root)
    dl_worker.download_references()
    dl_worker.download_metrics()
    dl_worker.download_captures()
    if include_binary:
        dl_worker.download_binary_files()
