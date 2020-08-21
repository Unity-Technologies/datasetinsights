import pytest

from datasetinsights.io.downloader.base import create_downloader
from datasetinsights.io.downloader.http_downloader import HTTPDatasetDownloader
from datasetinsights.io.downloader.unity_simulation import (
    UnitySimulationDownloader,
)


@pytest.mark.parametrize(
    "source_uri", ["http://", "https://"],
)
def test_create_downloader_http_downloader(source_uri):

    # act
    downloader = create_downloader(source_uri=source_uri)

    # assert
    assert isinstance(downloader, HTTPDatasetDownloader)


def test_create_downloader_unity_simulation_downloader():
    # arrange
    source_uri = "usim://"
    # act
    downloader = create_downloader(source_uri=source_uri)

    # assert
    assert isinstance(downloader, UnitySimulationDownloader)


def test_create_downloader_invalid_input():
    # arrange
    source_uri = "invalid_protocol://"
    # assert
    with pytest.raises(ValueError):
        # act
        create_downloader(source_uri=source_uri)


def test_create_downloader_none_input():
    # arrange
    source_uri = None
    # assert
    with pytest.raises(TypeError):
        # act
        create_downloader(source_uri=source_uri)
