from abc import ABCMeta, abstractmethod

from tensorboardX import SummaryWriter

import datasetinsights.constants as const
from datasetinsights.io.checkpoint import EstimatorCheckpoint
from datasetinsights.io.kfp_output import KubeflowPipelineWriter


def create_estimator(
    name,
    config,
    *,
    tb_log_dir=None,
    no_cuda=None,
    checkpoint_dir=None,
    kfp_metrics_dir=const.DEFAULT_KFP_METRICS_DIR,
    kfp_metrics_filename=const.DEFAULT_KFP_METRICS_FILENAME,
    no_val=None,
    **kwargs,
):
    """Create a new instance of the estimators subclass

    Args:
        name (str): unique identifier for a estimators subclass
        config (dict): parameters specific to each estimators subclass
            used to create a estimators instance

    Returns:
        an instance of the specified estimators subclass
    """

    estimators_cls = _find_estimator(name)

    # todo this makes it so that we lose the tensorboard
    #  writer of non-master processes which could make debugging harder

    writer = SummaryWriter(tb_log_dir)
    kfp_writer = KubeflowPipelineWriter(
        filename=kfp_metrics_filename, filepath=kfp_metrics_dir,
    )
    checkpointer = EstimatorCheckpoint(
        estimator_name=name, checkpoint_dir=checkpoint_dir, distributed=False,
    )

    return estimators_cls(
        config=config,
        writer=writer,
        kfp_writer=kfp_writer,
        checkpointer=checkpointer,
        logdir=tb_log_dir,
        no_cuda=no_cuda,
        no_val=no_val,
        kfp_metrics_dir=kfp_metrics_dir,
        kfp_metrics_filename=kfp_metrics_filename,
        **kwargs,
    )


def _find_estimator(name):
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


class Estimator(metaclass=ABCMeta):
    """Abstract base class for estimator.

    An estimator is the master class of all modeling operations. At minimum,
    it includes:

    1. input data and output data transformations (e.g. input image cropping,
    remove unused output labels...) when applicable.
    2. neural network graph (model) for either pytorch or tensorflow.
    3. procedures to execute model training and evaluation.

    One estimator could support multiple tasks (e.g. Mask R-CNN can be used for
    semantic segmentation and object detection)
    """

    @abstractmethod
    def train(self, **kwargs):
        """Abstract method to train estimators
        """
        raise NotImplementedError("Subclass needs to implement this method")

    @abstractmethod
    def evaluate(self, **kwargs):
        """Abstract method to evaluate estimators
        """
        raise NotImplementedError("Subclass needs to implement this method")
