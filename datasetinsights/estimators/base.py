from abc import ABCMeta, abstractmethod


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
