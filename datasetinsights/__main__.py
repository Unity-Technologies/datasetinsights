import logging

import click

from .configs import system
from .torch_distributed import get_world_size

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(levelname)s | %(asctime)s | %(name)s | %(threadName)s | "
        "%(message)s"
    ),
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@click.group()
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=system.verbose,
    help="Turn on verbose mode to enable debug logging.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=system.dryrun,
    help="Dry run with only a small subset of the dataset.",
)
@click.pass_context
def cli(ctx, verbose, dry_run):
    """Datasetinsights Interface"""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["dry_run"] = dry_run


# Train sub-command
@cli.command(
    short_help="Start Training",
    context_settings=dict(ignore_unknown_options=True, allow_extra_args=True),
)
@click.option(
    "--gpus-per-node",
    default=get_world_size(),
    help="Number of GPUs per node to use for training.",
)
@click.option(
    "--no-cuda/--cuda",
    default=system.no_cuda,
    help="Force to turn off CUDA training.",
)
@click.option(
    "--log-dir",
    type=str,
    default=system.logdir,
    help="Path for saving training logs.",
)
@click.option(
    "--metrics-dir",
    type=str,
    default=system.metricsdir,
    help="Path for saving metrics generated during training.",
)
@click.option(
    "--metrics-filename",
    type=str,
    default=system.metricsfilename,
    help="File name of metrics generated.",
)
@click.option(
    "-c",
    "--config",
    type=str,
    required=True,
    help="Config file for this " "model.",
)
@click.pass_context
def train(
    ctx, gpus_per_node, no_cuda, log_dir, metrics_dir, metrics_filename, config
):
    verbose = ctx.obj["verbose"]
    dry_run = ctx.obj["dry_run"]
    if verbose:
        click.echo("Training Started with Verbose")
    else:
        click.echo("Training Started")
    click.echo("Unknown Args: %s" % ctx.args)


# Evaluate sub-command
@cli.command(
    short_help="Start Evaluation",
    context_settings=dict(ignore_unknown_options=True, allow_extra_args=True),
)
@click.option(
    "--gpus-per-node",
    type=int,
    default=get_world_size(),
    help="Number of GPUs per node to use for evaluation.",
)
@click.option(
    "--no-cuda/--cuda",
    default=system.no_cuda,
    help="Force to turn off CUDA training.",
)
@click.option(
    "--log-dir",
    type=str,
    default=system.logdir,
    help="Path for saving evaluation logs",
)
@click.option(
    "--metrics-dir",
    type=str,
    default=system.metricsdir,
    help="Path for saving metrics generated during evaluation.",
)
@click.option(
    "--metrics-filename",
    type=str,
    default=system.metricsfilename,
    help="File name of metrics generated.",
)
@click.option(
    "-c",
    "--config",
    type=str,
    required=True,
    help="Config file for this " "model.",
)
@click.pass_context
def evaluate(
    ctx, gpus_per_node, no_cuda, log_dir, metrics_dir, metrics_filename, config
):
    verbose = ctx.obj["verbose"]
    dry_run = ctx.obj["dry_run"]
    click.echo("Evaluation Started")


# Download sub-command
@cli.command(
    short_help="Start Download",
    context_settings=dict(ignore_unknown_options=True, allow_extra_args=True),
)
@click.option(
    "--auth-token",
    type=str,
    default=system.auth_token,
    help="Authorization token to fetch USim manifest file that specified "
    "signed URLs for the simulation dataset files.",
)
@click.option(
    "--data-root",
    type=str,
    default=system.data_root,
    help="Root directory of all dataset",
)
@click.option(
    "--log-dir", type=str, default=system.logdir, help="Path for saving logs",
)
@click.pass_context
def download(ctx, log_dir, auth_token, data_root):
    verbose = ctx.obj["verbose"]
    dry_run = ctx.obj["dry_run"]
    click.echo("Downloading Data")


# cli call
cli(prog_name="python -m datasetinsights", obj={})
