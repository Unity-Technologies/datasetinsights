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


def init_distributed_mode(args):
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
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        logger.info(f"found 'SLURM_PROCID' in environment")
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        args.gpu = 0
        args.rank = 0
        logger.info("Not using distributed mode")
        args.distributed = False
        return
    device_count = torch.cuda.device_count()
    logger.info(f"device count: {torch.cuda.device_count()}")
    logger.info(f"world size: {args.world_size}")
    logger.info(f"gpu: {args.gpu}")
    logger.info(f"local rank {args.rank}")
    if device_count == 0:
        logger.info("No cuda devices found, will not parallelize")
        args.distributed = False
        return
    if not is_master():
        logging.disable(logging.ERROR)
    args.distributed = True
    torch.cuda.set_device(args.gpu)

    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.distributed.barrier()
