from abc import ABCMeta, abstractmethod


class DatasetRegistry(ABCMeta):
    registry = {}

    def __new__(cls, name, bases, namespace):

        if "name" not in namespace:
            raise RuntimeError(
                f"{name} must define a class-level dataset name!"
            )
        new_cls = super().__new__(cls, name, bases, namespace)
        if namespace["name"] != "":
            DatasetRegistry.registry[namespace["name"]] = new_cls
        return new_cls

    @classmethod
    def find(cls, name):
        dataset_cls = DatasetRegistry.registry.get(name)
        if dataset_cls:
            return dataset_cls
        else:
            raise ValueError(f"Dataset '{name}' does not exist:")

    @staticmethod
    def list_datasets():
        return DatasetRegistry.registry.keys()


class Dataset(metaclass=DatasetRegistry):
    """Abstract base class for datasets.
    """

    name = ""

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError("Subclass needs to implement this method")

    @abstractmethod
    def __len__(self):
        raise NotImplementedError("Subclass needs to implement this method")
