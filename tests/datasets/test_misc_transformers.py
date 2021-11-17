import tempfile

import pytest

from datasetinsights.datasets.transformers import (
    LSPETtoCOCOTransformer,
    MPIItoCOCOTransformer,
)


def test_lpset2coco_transformer_raise_value_error():
    with pytest.raises(ValueError):
        with tempfile.TemporaryDirectory() as tmp_dir:
            LSPETtoCOCOTransformer(data_root=tmp_dir)


def test_mpii2coco_transformer_raise_value_error():
    with pytest.raises(ValueError):
        with tempfile.TemporaryDirectory() as tmp_dir:
            MPIItoCOCOTransformer(data_root=tmp_dir)
