import re
from abc import ABCMeta, abstractmethod


class DownloaderRegistry(ABCMeta):
    registry = {}

    def __new__(cls, name, bases, namespace):
        protocal = "PROTOCAL"
        if protocal not in namespace:
            raise RuntimeError(f"{name} must define a class-level {protocal}")
        new_cls = super().__new__(cls, name, bases, namespace)
        if namespace[protocal] != "":
            DownloaderRegistry.registry[namespace[protocal]] = new_cls
        return new_cls

    @classmethod
    def find(cls, source_uri):
        match = re.compile("(gs://|^https://|^http://|^usim://)")
        protocal = match.findall(source_uri)[0]
        if protocal.startswith(("https://", "http://")):
            protocal = "http://"
        dataset_cls = DownloaderRegistry.registry.get(protocal)
        if dataset_cls:
            return dataset_cls
        else:
            raise ValueError(f"Downloader '{protocal}' does not exist:")

    @staticmethod
    def list_datasets():
        return DownloaderRegistry.registry.keys()


class DatasetDownloader(metaclass=DownloaderRegistry):
    PROTOCAL = ""

    @abstractmethod
    def download(self, **kwargs):

        raise NotImplementedError("Subclass needs to implement this method")
