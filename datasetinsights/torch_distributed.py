import os


def is_master():
    rank = int(os.getenv("RANK", 0))
    return rank == 0
