from abc import ABCMeta, abstractmethod


class DownloaderRegistry(ABCMeta):
    registry = {}

    def __new__(cls, name, bases, namespace):

        if "SOURCE_SCHEMA" not in namespace:
            raise RuntimeError(
                f"{name} must define a class-level SOURCE_SCHEMA"
            )
        new_cls = super().__new__(cls, name, bases, namespace)
        if namespace["SOURCE_SCHEMA"] != "":
            DownloaderRegistry.registry[namespace["SOURCE_SCHEMA"]] = new_cls
        return new_cls

    @classmethod
    def find(cls, source_uri_schema):
        dataset_cls = DownloaderRegistry.registry.get(source_uri_schema)
        if dataset_cls:
            return dataset_cls()
        else:
            raise ValueError(
                f"Downloader '{source_uri_schema}' does not exist:"
            )

    @staticmethod
    def list_datasets():
        return DownloaderRegistry.registry.keys()


class DatasetDownloader(metaclass=DownloaderRegistry):
    SOURCE_SCHEMA = ""

    @abstractmethod
    def download(self, **kwargs):

        raise NotImplementedError("Subclass needs to implement this method")


class Dataset:
    """Abstract base class for datasets.
    """

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError("Subclass needs to implement this method")

    @abstractmethod
    def __len__(self):
        raise NotImplementedError("Subclass needs to implement this method")
