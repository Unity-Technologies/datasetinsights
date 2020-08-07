import pytest
from click.exceptions import BadParameter

from datasetinsights.commands.download import SourceURI


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
