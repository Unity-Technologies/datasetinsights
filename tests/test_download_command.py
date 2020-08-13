from unittest.mock import patch

import pytest
from click.exceptions import BadParameter
from click.testing import CliRunner

from datasetinsights.commands.download import SourceURI, cli
from datasetinsights.datasets.base import DownloaderRegistry
from datasetinsights.datasets.unity_simulation import UnitySimulationDownloader


def test_source_uri_validation():
    validate_source_uri = SourceURI()

    gcs_path = "gs://bucket/path/to/folder"
    usim_path = "usim://auth@project_id/abdde"
    http_path = "http://domain/file.zip"
    https_path = "https://domain/file.zip"

    assert validate_source_uri(gcs_path) == gcs_path
    assert validate_source_uri(usim_path) == usim_path
    assert validate_source_uri(http_path) == http_path
    assert validate_source_uri(https_path) == https_path

    with pytest.raises(BadParameter):
        validate_source_uri("s3://bucket/file")
        validate_source_uri("/path/to/file")
        validate_source_uri("dasdklsdk")
        validate_source_uri("")


@pytest.mark.parametrize(
    "args",
    [
        ["download", "--source-uri=usim://", "--output=tests/"],
        ["download", "--source-uri=http://", "--output=tests/"],
        ["download", "--source-uri=https://", "--output=tests/"],
        ["download", "--source-uri=gs://", "--output=tests/"],
    ],
)
@patch.object(DownloaderRegistry, "find")
def test_download_except_called_once(mock_find, args):
    # arrange
    runner = CliRunner()
    # act
    runner.invoke(cli, args)
    # assert
    mock_find.assert_called_once()
    mock_find.return_value.return_value.download.assert_called_once()


@pytest.mark.parametrize(
    "args", [["download"], ["download", "--source-uri=s3://"]],
)
@patch.object(DownloaderRegistry, "find")
def test_download_except_not_called(mock_find, args):
    # arrange
    runner = CliRunner()
    # act
    runner.invoke(cli, args)
    # assert

    mock_find.assert_not_called()
    mock_find.return_value.download.assert_not_called()


def test_parsing_without_access_token_option():
    # arrange
    run_execution_id = "ABCDEDF"
    project_id = "aaaa-bbb-cccc-dddd-eeee"
    access_token = "access_token"
    downloader = UnitySimulationDownloader()

    # act
    downloader.parse_source_uri(
        f"usim://{access_token}@{project_id}/{run_execution_id}"
    )

    # assert
    assert downloader.run_execution_id == run_execution_id
    assert downloader.project_id == project_id
    assert downloader.access_token == access_token


def test_parsing_access_token_option():
    # arrange
    run_execution_id = "ABCDEDF"
    project_id = "aaaa-bbb-cccc-dddd-eeee"
    access_token = "access_token"
    downloader = UnitySimulationDownloader(access_token=access_token)

    # act
    downloader.parse_source_uri(f"usim://{project_id}/{run_execution_id}")

    # assert
    assert downloader.run_execution_id == run_execution_id
    assert downloader.project_id == project_id
    assert downloader.access_token == access_token


def test_parsing_access_token_option_and_source_uri_access_token():
    # arrange
    run_execution_id = "ABCDEDF"
    project_id = "aaaa-bbb-cccc-dddd-eeee"
    access_token = "access_token"
    downloader = UnitySimulationDownloader(access_token=access_token)

    # act
    downloader.parse_source_uri(
        f"usim://access_token_to_be_overridden@{project_id}/{run_execution_id}"
    )

    # assert
    assert downloader.run_execution_id == run_execution_id
    assert downloader.project_id == project_id
    assert downloader.access_token == access_token


@pytest.mark.parametrize("run_execution_id", ["ABCDEDF", "ABC-ABC"])
@pytest.mark.parametrize("project_id", ["invalid_project_id", "@abc", "@"])
@pytest.mark.parametrize("access_token", ["access_token", ""])
def test_download_with_invalid_source_uri(
    access_token, project_id, run_execution_id
):
    # arrange
    downloader = UnitySimulationDownloader(access_token=access_token)

    # assert
    with pytest.raises(ValueError):
        # act
        downloader.parse_source_uri(
            f"usim://access_token_to_be_overridden@{project_id}/{run_execution_id}"
        )
        downloader.parse_source_uri(
            f"usim://access_token_to_be_overridden{project_id}/{run_execution_id}"
        )
        downloader.parse_source_uri(f"usim://{project_id}/{run_execution_id}")
        downloader.parse_source_uri(
            f"usim://{access_token}@{project_id}/{run_execution_id}"
        )
        downloader.parse_source_uri(
            f"usim://{access_token}@@{project_id}/{run_execution_id}"
        )
