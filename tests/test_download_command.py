from unittest.mock import patch

import pytest
from click.exceptions import BadParameter
from click.testing import CliRunner

from datasetinsights.commands.download import SourceURI, cli


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
@patch("datasetinsights.commands.download.create_dataset_downloader")
def test_download_except_called_once(mock_create, args):
    # arrange
    runner = CliRunner()
    # act
    runner.invoke(cli, args)
    # assert
    mock_create.assert_called_once()
    mock_create.return_value.download.assert_called_once()


@pytest.mark.parametrize(
    "args",
    [["download"], ["download", "--source-uri=s3://"]],
)
@patch("datasetinsights.commands.download.create_dataset_downloader")
def test_download_except_not_called(mock_create, args):
    # arrange
    runner = CliRunner()
    # act
    runner.invoke(cli, args)
    # assert
    mock_create.assert_not_called()
    mock_create.return_value.download.assert_not_called()
