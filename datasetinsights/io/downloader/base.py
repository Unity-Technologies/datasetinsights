import re
from abc import ABC, abstractmethod

_registry = {}


class DownloaderFactory:
    """ This factory returns the correct DatasetDownloader
     from a registry based on the source-uri provided

    """

    @staticmethod
    def find(source_uri):
        """

        Args:
            source_uri: URI of where this data should be downloaded.

        Returns: The dataset downloader class that is
         registered with the source-uri protocol


        """
        protocols = "|".join(_registry.keys())
        pattern = re.compile(f"({protocols})")
        protocol = pattern.findall(source_uri)
        if protocol:
            protocol = protocol[0]
        else:
            raise ValueError(
                f"Downloader not found for source-uri '{source_uri}'"
            )

        if protocol.startswith(("https://", "http://")):
            protocol = "http://"

        return _registry.get(protocol)


class DatasetDownloader(ABC):
    @classmethod
    def __init_subclass__(cls, protocol=None, **kwargs):
        if protocol:
            _registry[protocol] = cls
        else:
            raise NotImplementedError(
                f"Subclass needs to have class keyword argument named protocol."
            )
        super().__init_subclass__(**kwargs)

    @abstractmethod
    def download(self, source_uri, output, **kwargs):
        raise NotImplementedError("Subclass needs to implement this method")
