import logging
import re

import click

import datasetinsights.constants as const
from datasetinsights.io import convert_synthetic_coco
from datasetinsights.io.downloader import GCSDatasetDownloader
from datasetinsights.io.downloader.base import create_dataset_downloader
from os import makedirs
from os.path import basename, isdir
import os
import zipfile

logger = logging.getLogger(__name__)


class SourceURI(click.ParamType):
    """Represents the Source URI Parameter type.

    This extends click.ParamType that allows click framework to validates
    supported source URI according to the prefix pattern.

    Raises:
        click.BadParameter: if the validation failed.
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


@click.command(context_settings=const.CONTEXT_SETTINGS,)
@click.option(
    "-s",
    "--source-uri",
    type=SourceURI(),
    required=True,
    help=(
        "URI of where this data should be downloaded. "
        f"Supported source uri patterns {SourceURI.PREFIX_PATTERN}"
    ),
)
@click.option(
    "-o",
    "--output",
    type=click.Path(exists=True, file_okay=False, writable=True),
    default=const.DEFAULT_DATA_ROOT,
    help="Directory on localhost where datasets should be downloaded.",
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
    "--access-token",
    type=str,
    default=None,
    help="Unity Simulation access token. "
    "This will override synthetic datasets source-uri for Unity Simulation",
)
@click.option(
    "--checksum-file",
    type=str,
    default=None,
    help="Dataset checksum text file path. "
    "Path can be a HTTP(S) url or a local file path. This will help check the "
    "integrity of the downloaded dataset.",
)
def cli(
    source_uri, output, include_binary, access_token, checksum_file,
):
    """Download datasets to localhost from known locations.

    The download command can support downloading from 3 types of sources

    1. Download from Unity Simulation:

    You can specify project_id, run_execution_id, access_token in source-uri:

    \b
    datasetinsights download \\
        --source-uri=usim://<access_token>@<project_id>/<run_execution_id> \\
        --output=$HOME/data

    Alternatively, you can also override access_token such as:

    \b
    datasetinsights download \\
        --source-uri=usim://<project_id>/<run_execution_id> \\
        --output=$HOME/data \\
        --access-token=<access_token>

    2. Downloading from a public http(s) url:

    \b
    datasetinsights download \\
        --source-uri=http://url/to/file.zip \\
        --output=$HOME/data

    3. Downloading from a GCS url:

    \b
    datasetinsights download \\
        --source-uri=gs://url/to/file.zip \\
        --output=$HOME/data

    or download all objects under the same directory:

    \b
    datasetinsights download \\
        --source-uri=gs://url/to/directory \\
        --output=$HOME/data
    """
    ctx = click.get_current_context()
    logger.debug(f"Called download command with parameters: {ctx.params}")

    # downloader = create_dataset_downloader(
    #     source_uri=source_uri, access_token=access_token
    # )
    # downloader.download(
    #     source_uri=source_uri,
    #     output=output,
    #     include_binary=include_binary,
    #     checksum_file=checksum_file,
    # )

    downloader = GCSDatasetDownloader()
    data_root = const.DEFAULT_DATA_ROOT
    if not isdir(data_root):
        makedirs(data_root)
    # data_root="/Users/sanjay.vishwakarma"
    datasets= source_uri.split("-")
    print(datasets)

    if "finetune" in source_uri:
        data_prefix="gs://live_reach/dataset/coco/"
        train_dir = data_root + "/coco_finetune"
        os.makedirs(train_dir, exist_ok=True)

        for i in range(1,len(datasets)):
            dataset_nm=datasets[i]
            downloader.download(
                source_uri=data_prefix + dataset_nm,
                output=train_dir,
            )
            if ".zip" in dataset_nm:
                with zipfile.ZipFile(train_dir+"/"+dataset_nm, "r") as zip_dir:
                    zip_dir.extractall(train_dir)

    else: # download synthetic train data
        data_prefix = "gs://live_reach/psp-20210729T1213/blobs/dataset/"
        train_dir = data_root + "/psp/train"
        os.makedirs(train_dir, exist_ok=True)
        for i in range(1, len(datasets)):
            dataset_nm = datasets[i]
            downloader.download(
                source_uri=data_prefix
                           + "peoplesanspeople-"
                           + dataset_nm,
                output=train_dir,
            )

        src_data = train_dir
        train_data = data_root + "/coco/train"
        os.makedirs(train_data, exist_ok=True)
        dataset_type = "train"

        convert_synthetic_coco(src_data, train_data, dataset_type, 0)

    #create val directory

    val_data = data_root + "/coco/val"
    os.makedirs(val_data, exist_ok=True)







