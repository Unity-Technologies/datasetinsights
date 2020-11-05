from torch.utils.tensorboard import SummaryWriter

from datasetinsights.torch_distributed import is_master


class DummySummaryWriter:
    """A fake summary writer that writes nothing to the disk. This writer is
    used when the process is not master process so that no data is written
    which can prevent overwriting real data. This writer mimics the
    SummaryWriter module in pytorch library. To see more about pytorch
    tensorbaord summary writer visit:
    https://github.com/pytorch/pytorch/blob/master/torch/utils/tensorboard/writer.py#L150
    """

    def __init__(self, log_dir, *args, **kwargs):
        self.logdir = log_dir

    def add_event(self, *args, **kwargs):
        return

    def add_summary(self, *args, **kwargs):
        return

    def add_graph(self, *args, **kwargs):
        return

    def add_scalar(self, *args, **kwargs):
        return

    def add_scalars(self, *args, **kwargs):
        return

    def add_histogram(self, *args, **kwargs):
        return

    def add_histogram_raw(self, *args, **kwargs):
        return

    def add_figure(self, *args, **kwargs):
        return

    def flush(self):
        return

    def close(self):
        return


def get_summary_writer():
    """
    Returns summary writer for tensorboard according to the process (master/
    non master)
    """
    if is_master():
        writer = SummaryWriter
    else:
        writer = DummySummaryWriter

    return writer
