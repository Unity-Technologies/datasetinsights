from abc import ABCMeta, abstractmethod


class Dataset(metaclass=ABCMeta):
    """Abstract base class for datasets.
    """

    @staticmethod
    def create(name, **kwargs):
        """Create a new instance of the dataset subclass

        Args:
            name (str): unique identifier for a dataset subclass
            config (dict): parameters specific to each dataset subclass
                used to create a dataset instance

        Returns:
            an instance of the specified dataset subclass
        """
        dataset_cls = Dataset.find(name)

        return dataset_cls(**kwargs)

    @staticmethod
    def find(name):
        """Find Dataset subclass based on the given name

        Args:
            name (str): unique identifier for a dataset subclass

        Returns:
            a label of the specified dataset subclass
        """
        dataset_classes = Dataset.__subclasses__()
        dataset_names = [d.__name__ for d in dataset_classes]
        if name in dataset_names:
            dataset_cls = dataset_classes[dataset_names.index(name)]
            return dataset_cls
        else:
            raise NotImplementedError(f"Unknown Dataset class {name}!")

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError("Subclass needs to implement this method")

    @abstractmethod
    def __len__(self):
        raise NotImplementedError("Subclass needs to implement this method")
