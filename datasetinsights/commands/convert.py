import logging

import click

import datasetinsights.constants as const
from datasetinsights.datasets.transformers import get_dataset_transformer

logger = logging.getLogger(__name__)


@click.command(context_settings=const.CONTEXT_SETTINGS)
@click.option(
    "-i",
    "--input",
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help="Directory of the dataset to be converted.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(file_okay=False, writable=True),
    required=True,
    help="Directory of the converted dataset.",
)
@click.option(
    "-f",
    "--format",
    required=True,
    help=(
        "The output dataset format. "
        "Currently only 'COCO-Instances' and 'COCO-Keypoints' is supported."
    ),
)
@click.option(
    "-d",
    "--dbtype",
    type=click.Path(exists=True, file_okay=False),
    help="Type of converted dataset. Can be 'train', 'val' or 'test'",
)
@click.option(
    "--ann-file-path",
    type=click.Path(exists=True, file_okay=False),
    help="Path of annotation file of the dataset to be converted.",
)
def cli(input, output, format, db_type, ann_file_path):
    """Convert dataset from Perception format to target format.
    """
    ctx = click.get_current_context()
    logger.debug(f"Called convert command with parameters: {ctx.params}")

    transformer = get_dataset_transformer(
        format=format, input=input, db_type=db_type, ann_file_path=ann_file_path
    )
    transformer.execute(output=output)
