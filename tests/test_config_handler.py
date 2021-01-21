from unittest.mock import MagicMock, patch

import pytest
from yacs.config import CfgNode as CN

from datasetinsights.io.config_handler import (
    load_config,
    override_config,
    prepare_config,
)


@pytest.fixture
def config():
    """prepare config."""
    with open("tests/configs/faster_rcnn_groceries_real_test.yaml") as f:
        cfg = CN.load_cfg(f)

    return cfg


def test_override_config(config):
    actual_value = True
    expected_value = False
    override = f"pretrained={expected_value}"
    config.pretrained = actual_value
    returned_config = override_config(override=override, config=config)
    assert returned_config["pretrained"] != actual_value
    assert returned_config["pretrained"] == expected_value


def test_invalid_override_key(config):
    override = "a.b.c=test1 c.a"
    with pytest.raises(AssertionError):
        override_config(override=override, config=config)


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
