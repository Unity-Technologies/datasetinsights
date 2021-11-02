from abc import ABC, abstractmethod

_registry = {}


def get_dataset_transformer(format, **kwargs):
    """
    Returns instantiated transformer object based on the provided conversion
    format from a registry.

    Args:
        format (str): Conversion format to be used for dataset transformation.

    Returns: Transformer object instance.

    """
    if format in _registry.keys():
        transformer = _registry[format]
    else:
        raise ValueError(f"Transformer not found for conversion format "
                         f"'{format}'")

    return transformer(**kwargs)


class DatasetTransformer(ABC):
    """ Base class for all ddataset transformer.
    """

    def __init__(self, **kwargs):
        pass

    @classmethod
    def __init_subclass__(cls, format=None, **kwargs):
        if format:
            _registry[format] = cls
        else:
            raise NotImplementedError(
                f"Subclass needs to have class keyword argument named "
                f"transformer."
            )
        super().__init_subclass__(**kwargs)

    @abstractmethod
    def execute(self, output, **kwargs):
        raise NotImplementedError("Subclass needs to implement this method")
