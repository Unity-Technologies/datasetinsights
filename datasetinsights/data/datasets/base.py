from abc import ABCMeta, abstractmethod


class Downloader(metaclass=ABCMeta):
    """Abstract base class for dataset downloaders.

    The following is necessary for a valid downloader class:

    - It needs a class variable 'NAME'.
    - It needs a class variable 'SOURCE_URI_SCHEMA' such as http:// or gs://.
    - It needs to define a `download` instance method.
    - It should provide proper authenticaltion/authorization mechanism
        before the download method is called.
    """

    REQUIRED_SUBCLASS_VARIABLES = ["NAME", "SOURCE_URI_SCHEMA"]

    @staticmethod
    def create(name, **kwargs):
        """ Create a new instance of the dataset downloader.

        Args:
            name (str): Unique identifier for a dataset downloader subclass.
            source_uri

        Returns:
            Downloader: An instance of the specified dataset downloader
                subclass.
        """
        downloader_cls = Downloader.find(name)

        return downloader_cls(**kwargs)

    @staticmethod
    def find(name):
        """ Find Dataset downloader subclass based on the given name.

        Args:
            name (str): Unique identifier for a dataset downloader subclass.

        Returns:
            Downloader: A label of the specified dataset downloader subclass.
        """
        downloaders = Downloader.__subclasses__()
        downloader_names = [d.NAME for d in downloaders]
        if name in downloader_names:
            downloader_cls = downloader_names[downloader_names.index(name)]
            return downloader_cls
        else:
            raise NotImplementedError(
                f"Unknown dataset downloader name: {name}!"
                f"Supported downloader names are: {downloader_names}"
            )

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

    @abstractmethod
    def download(self, **kwargs):
        raise NotImplementedError("Subclass needs to implement this method.")


class Dataset(metaclass=ABCMeta):
    """Abstract base class for datasets.
    """

    @staticmethod
    def create(name, **kwargs):
        """Create a new instance of the dataset subclass

        Args:
            name (str): unique identifier for a dataset subclass

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
