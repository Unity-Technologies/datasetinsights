from abc import abstractmethod


class DownloaderRegistry:
    """ The Registry class for datasets"""

    dataset_registry = {}
    download_registry = {}
    """ Internal registry for available Datasets """

    @classmethod
    def register(cls, name: str, downloaders: list):
        """ Class method to register Dataset class to the internal registry.
        Args:
            name (str): The name of the Dataset.
        Returns:
            The Dataset class itself.
        """

        def inner_wrapper(wrapped_class: Dataset):
            if name in cls.dataset_registry:
                raise RuntimeError(f"Dataset name already registered")
            cls.dataset_registry[name] = wrapped_class
            cls.download_registry[wrapped_class] = downloaders
            return wrapped_class

        return inner_wrapper

    @classmethod
    def find(cls, name, source_uri=None):
        dataset_cls = DownloaderRegistry.dataset_registry.get(name)
        if dataset_cls:
            downloaders = DownloaderRegistry.download_registry.get(dataset_cls)
            for downloader in downloaders:
                if source_uri and source_uri.startswith(
                    downloader.SOURCE_URI_SCHEMA
                ):
                    return downloader
            return downloaders[0]
        else:
            raise ValueError(f"Dataset '{name}' does not exist:")

    @staticmethod
    def list_datasets():
        return DownloaderRegistry.dataset_registry.keys()


class DatasetDownloader:
    REQUIRED_SUBCLASS_VARIABLES = ["SOURCE_URI_SCHEMA"]

    @abstractmethod
    def download(self, **kwargs):

        raise NotImplementedError("Subclass needs to implement this method")

    @classmethod
    def __init_subclass__(cls):
        """ Executed during subclass init.
        Raises:
            NotImplementedError: Raised if the subclass does not define
                required variables.
        """
        for var in cls.REQUIRED_SUBCLASS_VARIABLES:
            if not hasattr(cls, var):
                raise NotImplementedError(
                    f"Subclass needs to have a constant class variable '{var}'."
                )


class Dataset:
    """Abstract base class for datasets.
    """

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError("Subclass needs to implement this method")

    @abstractmethod
    def __len__(self):
        raise NotImplementedError("Subclass needs to implement this method")
