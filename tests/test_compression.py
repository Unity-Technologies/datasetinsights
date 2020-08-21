import os
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from datasetinsights.io.compression import Compression


def test_compression_factory_returns_zipfile_compression():
    with tempfile.NamedTemporaryFile() as tmp:
        with ZipFile(tmp, "w", ZIP_DEFLATED) as archive:
            archive.writestr("something.txt", "Some Content Here")
        Compression.decompress(filepath=archive.filename, destination=tmp.name)
        assert os.path.isfile(os.path.join(tmp.name, "something.txt"))


@pytest.mark.parametrize(
    "suffix", [".txt", ".zip", ".txt.gz", ".tar", ".tar.gz"],
)
def test_compression_factory_raises_value_error(suffix):
    with tempfile.NamedTemporaryFile(prefix="file", suffix=suffix) as tmp:
        with pytest.raises(ValueError):
            Compression.decompress(filepath=tmp.name, destination=tmp)
