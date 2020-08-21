import gzip
import tarfile
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from datasetinsights.io.compression import (
    GZipCompression,
    TarFileCompression,
    ZipFileCompression,
    compression_factory,
)


def test_compression_factory_returns_zipfile_compression():
    with tempfile.NamedTemporaryFile() as tmp:
        with ZipFile(tmp, "w", ZIP_DEFLATED) as archive:
            archive.writestr("something.txt", "Some Content Here")
        assert (
            compression_factory(filepath=archive.filename) == ZipFileCompression
        )


def test_compression_factory_returns_tarfile_compression():
    with tempfile.NamedTemporaryFile() as tmp:
        with tempfile.NamedTemporaryFile() as tmp2:
            with tarfile.TarFile(tmp.name, "w") as archive:
                archive.add(tmp2.name)

            assert compression_factory(filepath=tmp.name) == TarFileCompression


def test_compression_factory_returns_gzip_compression():
    with tempfile.NamedTemporaryFile() as tmp:
        with gzip.GzipFile(tmp.name, "w") as archive:
            archive.write(b"Some Content Here")
        assert compression_factory(filepath=tmp.name) == GZipCompression


@pytest.mark.parametrize(
    "suffix", [".txt", ".zip", ".txt.gz", ".tar", ".tar.gz"],
)
def test_compression_factory_raises_value_error(suffix):
    with tempfile.NamedTemporaryFile(prefix="file", suffix=suffix) as tmp:
        with pytest.raises(ValueError):
            compression_factory(filepath=tmp.name)
