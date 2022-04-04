import pytest

from datasetinsights.io.downloader.base import create_dataset_downloader
from datasetinsights.io.downloader.http_downloader import HTTPDatasetDownloader


@pytest.mark.parametrize(
    "source_uri",
    ["http://", "https://"],
)
def test_create_dataset_downloader_http_downloader(source_uri):

    # act
    downloader = create_dataset_downloader(source_uri=source_uri)

    # assert
    assert isinstance(downloader, HTTPDatasetDownloader)


def test_create_dataset_downloader_invalid_input():
    # arrange
    source_uri = "invalid_protocol://"
    # assert
    with pytest.raises(ValueError):
        # act
        create_dataset_downloader(source_uri=source_uri)


def test_create_dataset_downloader_none_input():
    # arrange
    source_uri = None
    # assert
    with pytest.raises(TypeError):
        # act
        create_dataset_downloader(source_uri=source_uri)
