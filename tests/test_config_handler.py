from unittest.mock import MagicMock, patch

import pytest

from datasetinsights.io.config_handler import load_config


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

    config_url = "http://thea-dev/test/config.yaml"
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

    config_url = "https://thea-dev/test/config.yaml"
    load_config(config_url)

    mock_config.assert_called_with(mock_obj)
    mock_create_downloader.assert_called_with(source_uri=config_url)
    mock_obj.download.assert_called_once()


@patch("datasetinsights.io.config_handler.CfgNode.load_cfg")
@patch("datasetinsights.io.config_handler.open")
def test_load_config_file_prefix(mock_open, mock_config):
    mock_obj = MagicMock()
    mock_open.return_value = mock_obj
    config_url = "file:///root/test.yaml"
    load_config(config_url)
    mock_config.assert_called_with(mock_obj)


@patch("datasetinsights.io.config_handler.CfgNode.load_cfg")
@patch("datasetinsights.io.config_handler.open")
def test_load_config_absolute_path(mock_open, mock_config):
    mock_obj = MagicMock()
    mock_open.return_value = mock_obj
    config_url = "/root/test.yaml"
    load_config(config_url)
    mock_config.assert_called_with(mock_obj)


@patch("datasetinsights.io.config_handler.CfgNode.load_cfg")
@patch("datasetinsights.io.config_handler.open")
def test_load_config_relative_path(mock_open, mock_config):
    mock_obj = MagicMock()
    mock_open.return_value = mock_obj
    config_url = "datasetinsights/config.yaml"
    load_config(config_url)
    mock_config.assert_called_with(mock_obj)


def test_bad_path():
    bad_url = "s3://path/to/bad/url"
    with pytest.raises(ValueError):
        load_config(bad_url)
