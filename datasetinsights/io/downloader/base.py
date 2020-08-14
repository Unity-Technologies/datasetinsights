import re
from abc import ABCMeta, abstractmethod


class DownloaderRegistry(ABCMeta):
    """ This class registers DatasetDownloader
        subclasses based on the PROTOCOL attribute

    """

    registry = {}

    def __new__(cls, name, bases, namespace):
        protocol = "PROTOCOL"
        if protocol not in namespace:
            raise RuntimeError(f"{name} must define a class-level {protocol}")
        new_cls = super().__new__(cls, name, bases, namespace)
        if namespace[protocol] != "":
            DownloaderRegistry.registry[namespace[protocol]] = new_cls
        return new_cls

    @classmethod
    def find(cls, source_uri):
        match = re.compile("(gs://|^https://|^http://|^usim://)")
        protocol = match.findall(source_uri)[0]
        if protocol.startswith(("https://", "http://")):
            protocol = "http://"
        dataset_cls = DownloaderRegistry.registry.get(protocol)
        if dataset_cls:
            return dataset_cls
        else:
            raise ValueError(f"Downloader '{protocol}' does not exist:")

    @staticmethod
    def list_datasets():
        return DownloaderRegistry.registry.keys()


class DatasetDownloader(metaclass=DownloaderRegistry):
    PROTOCOL = ""

    @abstractmethod
    def download(self, **kwargs):

        raise NotImplementedError("Subclass needs to implement this method")
