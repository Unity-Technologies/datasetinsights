import tempfile

import pytest

from datasetinsights.io.compression import decompress


@pytest.mark.parametrize(
    "suffix", [".txt", ".zip", ".txt.gz", ".tar", ".tar.gz"],
)
def test_decompress_raises_value_error(suffix):
    with tempfile.NamedTemporaryFile(prefix="file", suffix=suffix) as tmp:
        with pytest.raises(ValueError):
            decompress(filepath=tmp.name, destination=tmp)
