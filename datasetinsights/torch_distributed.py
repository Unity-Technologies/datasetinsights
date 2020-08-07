import logging
import os

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def is_master():
    rank = int(os.getenv("RANK", 0))
    return rank == 0


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    """

    Returns: number of available devices

    """
    if not is_dist_avail_and_initialized():
        return 1
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    return dist.get_world_size()


def get_rank():
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        return rank
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
    else:
        return 0


def get_gpu():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        gpu = rank % torch.cuda.device_count()
        return gpu
    else:
        return 0


def init_distributed_mode():
    """
    This method assumes that the module has been launched using
    torch.distributed.launch which sets the proper environment variables.
    It parses those variables and initializes the process group.
    https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py
    """
    if not is_master():
        logging.disable(logging.ERROR)

    if torch.cuda.device_count() == 0:
        logger.info("No cuda devices found, will not parallelize")
        return
    if (
        "RANK" not in os.environ
        and "WORLD_SIZE" not in os.environ
        or "SLURM_PROCID" not in os.environ
    ):
        return

    gpu = get_gpu()
    rank = get_rank()
    world_size = get_world_size()

    logger.info(f"gpu: {gpu}")
    logger.info(f"local rank {rank}")
    logger.info(f"world size {world_size}")

    torch.cuda.set_device(gpu)
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://", world_size=world_size, rank=rank,
    )
    torch.distributed.barrier()


def is_distributed():
    if (
        "RANK" not in os.environ
        and "WORLD_SIZE" not in os.environ
        or "SLURM_PROCID" not in os.environ
    ):
        logger.info("Not using distributed mode")
        return False
    if torch.cuda.device_count() == 0:
        logger.info("No cuda devices found, will not parallelize")
        return False
    else:
        return True
