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
    return dist.get_world_size()


def init_distributed_mode():
    """
    This method assumes that the module has been launched using
    torch.distributed.launch which sets the proper environment variables.
    It parses those variables and initializes the process group.
    https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py
    Args:
        args: cli arguments

    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        logger.info(f"found RANK and WORLD_SIZE in environment")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        logger.info(f"found 'SLURM_PROCID' in environment")
        rank = int(os.environ["SLURM_PROCID"])
        gpu = rank % torch.cuda.device_count()
    else:
        gpu = 0
        rank = 0
        logger.info("Not using distributed mode")
        distributed = False
        return gpu, rank, distributed
    device_count = torch.cuda.device_count()
    logger.info(f"device count: {torch.cuda.device_count()}")
    logger.info(f"world size: {world_size}")
    logger.info(f"gpu: {gpu}")
    logger.info(f"local rank {rank}")
    if device_count == 0:
        logger.info("No cuda devices found, will not parallelize")
        distributed = False
        return gpu, rank, distributed
    if not is_master():
        logging.disable(logging.ERROR)
    distributed = True
    torch.cuda.set_device(gpu)

    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )
    torch.distributed.barrier()
    return gpu, rank, distributed
