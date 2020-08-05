from abc import ABCMeta, abstractmethod


class EvaluationMetric(metaclass=ABCMeta):
    """Abstract base class for metrics.
    """

    @staticmethod
    def create(name, **kwargs):
        """Create a new instance of the metric subclass

        Args:
            name (str): unique identifier for a metric subclass
            config (dict): parameters specific to each metric subclass
                used to create a metric instance

        Returns:
            an instance of the specified metric subclass
        """
        metric_cls = EvaluationMetric.find(name)

        return metric_cls(**kwargs)

    @staticmethod
    def find(name):
        """Find EvaluationMetric subclass based on the given name

        Args:
            name (str): unique identifier for a metric subclass

        Returns:
            a label of the specified metric subclass
        """
        metric_classes = EvaluationMetric.get_all_subclasses(EvaluationMetric)
        metric_names = [d.__name__ for d in metric_classes]
        if name in metric_names:
            metric_cls = metric_classes[metric_names.index(name)]
            return metric_cls
        else:
            raise NotImplementedError(f"Unknown Metric class {name}!")

    @staticmethod
    def get_all_subclasses(cls):
        """Find EvaluationMetric all subclasses, subclasses of subclasses, and so on
        """
        all_subclasses = []

        for subclass in cls.__subclasses__():
            all_subclasses.append(subclass)
            all_subclasses.extend(EvaluationMetric.get_all_subclasses(subclass))

        return all_subclasses

    @abstractmethod
    def reset(self):
        raise NotImplementedError("Subclass needs to implement this method")

    @abstractmethod
    def update(self, output):
        raise NotImplementedError("Subclass needs to implement this method")

    @abstractmethod
    def compute(self):
        raise NotImplementedError("Subclass needs to implement this method")
