from abc import ABC, abstractmethod


def get_dataset_transformer(format, **kwargs):
    """
    Returns instantiated transformer object based on the provided conversion
    format from a registry.

    Args:
        format (str): Conversion format to be used for dataset transformation.

    Returns: Transformer object instance.

    """
    if format in DatasetTransformer.REGISTRY.keys():
        transformer = DatasetTransformer.REGISTRY[format]
    else:
        raise ValueError(
            f"Transformer not found for conversion format '{format}'"
        )

    return transformer(**kwargs)


class DatasetTransformer(ABC):
    """Base class for all dataset transformer."""

    REGISTRY = {}

    @classmethod
    def __init_subclass__(cls, format=None, **kwargs):
        if format:
            cls.REGISTRY[format] = cls
        else:
            raise NotImplementedError(
                f"Subclass needs to have class keyword argument named "
                f"transformer."
            )
        super().__init_subclass__(**kwargs)

    @abstractmethod
    def execute(self, output, **kwargs):
        raise NotImplementedError("Subclass needs to implement this method")
