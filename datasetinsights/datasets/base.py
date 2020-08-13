from abc import abstractmethod


class Dataset:
    """Abstract base class for datasets.
    """

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError("Subclass needs to implement this method")

    @abstractmethod
    def __len__(self):
        raise NotImplementedError("Subclass needs to implement this method")
