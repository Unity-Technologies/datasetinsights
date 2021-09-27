import logging

import click

import datasetinsights.constants as const
from datasetinsights.datasets.transformers import COCOInstancesTransformer

logger = logging.getLogger(__name__)


@click.command(context_settings=const.CONTEXT_SETTINGS)
@click.option(
    "-i",
    "--input",
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help="Directory of the Synthetic dataset.",
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
        "Currently only 'COCO-Instances' is supported."
    ),
)
def cli(input, output, format):
    """Convert dataset from Perception format to target format.
    """
    ctx = click.get_current_context()
    logger.debug(f"Called convert command with parameters: {ctx.params}")

    # TODO(YC) support other formats like COCO-Keypoints
    if format != "COCO-Instances":
        raise ValueError(f"Unsupported target conversion format {format}")
    transformer = COCOInstancesTransformer(input)
    transformer.execute(output)
