import pytest

from datasetinsights.io.downloader.base import create_downloader
from datasetinsights.io.downloader.http_downloader import HTTPDownloader
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
    assert isinstance(downloader, HTTPDownloader)


def test_create_downloader_unity_simulation_downloader():
    # arrange
    source_uri = "usim://"
    # act
    downloader = create_downloader(source_uri=source_uri)

    # assert
    assert isinstance(downloader, UnitySimulationDownloader)
