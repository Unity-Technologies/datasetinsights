import logging

import click

import datasetinsights.constants as const
from datasetinsights.datasets.transformers import COCOTransformer

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
    help=("The output dataset format. Currently only COCO is supported."),
)
# TODO(YC) We need to figure out a better way to supply definition ID here
# in order to support other annotations like instance segmentation or keypoints.
# Ideally, we should be able to detect all available annotations that's
# compatible to COCO formats.
@click.option(
    "-d",
    "--bbox2d-definition-id",
    required=True,
    help=("The 2D bounding box annotation definition ID"),
)
def cli(input, output, format, bbox2d_definition_id):
    """Convert dataset from Perception format to target format.
    """
    ctx = click.get_current_context()
    logger.debug(f"Called convert command with parameters: {ctx.params}")

    # TODO(YC) support other types of datasets
    if format != "COCO":
        raise ValueError(f"Unsupported target conversion format {format}")
    transformer = COCOTransformer(input, bbox2d_definition_id)
    transformer.execute(output)
