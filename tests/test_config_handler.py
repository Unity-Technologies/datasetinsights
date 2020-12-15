from unittest.mock import MagicMock, patch

import pytest
from yaml.scanner import ScannerError

from datasetinsights.io.config_handler import (
    load_config,
    override_config,
    prepare_config,
)
from datasetinsights.io.exceptions import (
    ConfigLoadError,
    DownloadError,
    InvalidOverrideKey,
)


@patch("datasetinsights.io.config_handler.create_downloader")
@patch("datasetinsights.io.config_handler.CfgNode.load_cfg")
@patch("datasetinsights.io.config_handler.open")
def test_gcs_load_config(mock_open, mock_config, mock_create_downloader):
    mock_obj = MagicMock()
    mock_create_downloader.return_value = mock_obj
    mock_open.return_value = mock_obj

    config_url = "gs://test/test.yaml"
    load_config(config_url)

    mock_config.assert_called_with(mock_obj)
    mock_create_downloader.assert_called_with(source_uri=config_url)
    mock_obj.download.assert_called_once()


@patch("datasetinsights.io.config_handler.create_downloader")
@patch("datasetinsights.io.config_handler.CfgNode.load_cfg")
@patch("datasetinsights.io.config_handler.open")
def test_http_load_config(mock_open, mock_config, mock_create_downloader):
    mock_obj = MagicMock()
    mock_create_downloader.return_value = mock_obj
    mock_open.return_value = mock_obj

    config_url = "http://thea-dev/../config.yaml"
    load_config(config_url)

    mock_config.assert_called_with(mock_obj)
    mock_create_downloader.assert_called_with(source_uri=config_url)
    mock_obj.download.assert_called_once()


@patch("datasetinsights.io.config_handler.create_downloader")
@patch("datasetinsights.io.config_handler.CfgNode.load_cfg")
@patch("datasetinsights.io.config_handler.open")
def test_https_load_config(mock_open, mock_config, mock_create_downloader):
    mock_obj = MagicMock()
    mock_create_downloader.return_value = mock_obj
    mock_open.return_value = mock_obj

    config_url = "https://thea-dev/../config.yaml"
    load_config(config_url)

    mock_config.assert_called_with(mock_obj)
    mock_create_downloader.assert_called_with(source_uri=config_url)
    mock_obj.download.assert_called_once()


@patch("datasetinsights.io.config_handler.CfgNode.load_cfg")
@patch("datasetinsights.io.config_handler.open")
def test_local_load_config(mock_open, mock_config):
    mock_obj = MagicMock()
    mock_open.return_value = mock_obj

    config_url = "file://test.yaml"
    load_config(config_url)
    mock_config.assert_called_with(mock_obj)


@patch("datasetinsights.io.config_handler.override_config")
@patch("datasetinsights.io.config_handler.load_config")
def test_prepare_config(mock_load, mock_override):
    mock_obj = MagicMock()
    mock_load.return_value = mock_obj
    config_url = "test.yaml"
    override = "test=test"
    prepare_config(path=config_url, override=override)
    mock_load.assert_called_with(config_url)
    mock_override.assert_called_with(override=override, config=mock_obj)


@patch("datasetinsights.io.config_handler.create_downloader")
def test_bad_path(mock_create_downloader):
    mock_create_downloader.side_effect = ValueError
    bad_url = "s3://path/to/bad/url"
    with pytest.raises(ValueError):
        load_config(bad_url)


@patch("datasetinsights.io.config_handler.create_downloader")
def test_download_fail(mock_create_downloader):
    mock_create_downloader.download.side_effect = DownloadError
    bad_url = "http://path/to/bad/url"
    with pytest.raises(ConfigLoadError):
        load_config(bad_url)


@patch("datasetinsights.io.config_handler.create_downloader")
@patch("datasetinsights.io.config_handler.CfgNode.load_cfg")
def test_load_fail(mock_load, mock_create_downloader):
    mock_load.side_effect = ScannerError
    bad_url = "http://path/to/bad/url"
    with pytest.raises(ConfigLoadError):
        load_config(bad_url)


@patch("datasetinsights.io.config_handler.CfgNode")
def test_override_config(mock_cfg):

    override = "a.b.c=test1 c.a=test2"
    override_token = ["a.b.c", "test1", "c.a", "test2"]
    override_config(override=override, config=mock_cfg)
    mock_cfg.merge_from_list.assert_called_with(override_token)


@patch("datasetinsights.io.config_handler.CfgNode")
def test_invalid_override_key(mock_cfg):
    mock_cfg.merge_from_list.side_effect = AssertionError
    override = "a.b.c=test1 c.a"
    with pytest.raises(InvalidOverrideKey):
        override_config(override=override, config=mock_cfg)
