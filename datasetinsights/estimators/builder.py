import torch
from tensorboardX import SummaryWriter

import datasetinsights.constants as const
from datasetinsights.estimators import Estimator
from datasetinsights.estimators.faster_rcnn import FasterRCNNDepedencies
from datasetinsights.storage.checkpoint import EstimatorCheckpoint
from datasetinsights.storage.kfp_output import KubeflowPipelineWriter
from datasetinsights.torch_distributed import init_distributed_mode, is_master


class EstimatorBuilder:
    @staticmethod
    def create(name, **kwargs):
        """Create a new instance of the estimators subclass

        Args:
            name (str): unique identifier for a estimators subclass
            config (dict): parameters specific to each estimators subclass
                used to create a estimators instance

        Returns:
            an instance of the specified estimators subclass
        """

        estimators_cls = EstimatorBuilder.find(name)
        estimator_dependencies = create_estomator_depedencies(
            kwargs["ctx"], kwargs["model_config"]
        )

        return estimators_cls(
            estimator_dependencies=estimator_dependencies, **kwargs
        )

    @staticmethod
    def find(name):
        """Find Estimator subclass based on the given name

        Args:
            name (str): unique identifier for a estimators subclass

        Returns:
            a label of the specified estimators subclass
        """
        estimators_classes = Estimator.__subclasses__()
        estimators_names = [e.__name__ for e in estimators_classes]
        if name in estimators_names:
            estimators_cls = estimators_classes[estimators_names.index(name)]
            return estimators_cls
        else:
            raise NotImplementedError(f"Unknown Estimator class {name}!")


def create_estomator_depedencies(ctx, model_config):

    if len(ctx.args) == 1:
        model_config.merge_from_list(ctx.args[0].split(" "))
    else:
        model_config.merge_from_list(ctx.args)

    gpu, rank, distributed = init_distributed_mode()
    if torch.cuda.is_available() and not ctx.params["no_cuda"]:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logdir = ctx.params["tb_log_dir"]
    if logdir == const.NULL_STRING:
        # Use logdir=None to force using SummaryWriter default logdir,
        # which points to ./runs/<model>_<timestamp>
        logdir = None

    # todo this makes it so that we lose the tensorboard
    #  writer of non-master processes which could make debugging harder
    writer = SummaryWriter(
        logdir,
        write_to_disk=is_master(),
        max_queue=const.SUMMARY_WRITER_MAX_QUEUE,
        flush_secs=const.SUMMARY_WRITER_FLUSH_SECS,
    )
    kfp_writer = KubeflowPipelineWriter(
        filename=const.DEFAULT_KFP_METRICS_FILENAME,
        filepath=const.DEFAULT_KFP_METRICS_DIR,
    )
    checkpointer = EstimatorCheckpoint(
        estimator_name=model_config.estimator,
        log_dir=writer.logdir,
        distributed=distributed,
    )
    estimator_depedencies = FasterRCNNDepedencies(
        config=model_config,
        writer=writer,
        kfp_writer=kfp_writer,
        device=device,
        checkpointer=checkpointer,
        gpu=gpu,
        rank=rank,
        distributed=distributed,
        data_root=ctx.params["data_root"],
    )

    return estimator_depedencies
