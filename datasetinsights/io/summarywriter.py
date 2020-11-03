class DummySummaryWriter:
    """A fake summary writer that writes nothing to the disk.
    """

    def add_event(self, event, step=None, walltime=None):
        return

    def add_summary(self, summary, global_step=None, walltime=None):
        return

    def add_graph(self, graph_profile, walltime=None):
        return

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        return

    def add_scalars(
        self, main_tag, tag_scalar_dict, global_step=None, walltime=None
    ):
        return

    def add_histogram(
        self,
        tag,
        values,
        global_step=None,
        bins="tensorflow",
        walltime=None,
        max_bins=None,
    ):
        return

    def add_histogram_raw(
        self,
        tag,
        min,
        max,
        num,
        sum,
        sum_squares,
        bucket_limits,
        bucket_counts,
        global_step=None,
        walltime=None,
    ):
        return

    def add_figure(
        self, tag, figure, global_step=None, close=True, walltime=None
    ):
        return

    def flush(self):
        return

    def close(self):
        return
