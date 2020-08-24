import os

import torch.distributed as dist


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
