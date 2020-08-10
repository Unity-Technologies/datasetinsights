from tensorboardX import SummaryWriter

import datasetinsights.constants as const
from datasetinsights.estimators import Estimator
from datasetinsights.storage.checkpoint import EstimatorCheckpoint
from datasetinsights.storage.kfp_output import KubeflowPipelineWriter


class EstimatorFactory:
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

        estimators_cls = EstimatorFactory.find(name)

        logdir = kwargs["params"]["tb_log_dir"]
        no_cuda = kwargs["params"]["no_cuda"]
        data_root = kwargs["params"]["data_root"]
        model_config = kwargs["model_config"]

        if logdir == const.NULL_STRING:
            # Use logdir=None to force using SummaryWriter default logdir,
            # which points to ./runs/<model>_<timestamp>
            logdir = None

        # todo this makes it so that we lose the tensorboard
        #  writer of non-master processes which could make debugging harder

        writer = SummaryWriter(logdir,)
        kfp_writer = KubeflowPipelineWriter(
            filename=const.DEFAULT_KFP_METRICS_FILENAME,
            filepath=const.DEFAULT_KFP_METRICS_DIR,
        )
        checkpointer = EstimatorCheckpoint(
            estimator_name=model_config.estimator,
            log_dir=writer.logdir,
            distributed=False,
        )

        return estimators_cls(
            config=model_config,
            writer=writer,
            kfp_writer=kfp_writer,
            checkpointer=checkpointer,
            logdir=logdir,
            data_root=data_root,
            no_cuda=no_cuda,
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
