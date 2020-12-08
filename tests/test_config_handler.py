from unittest.mock import MagicMock, patch

from datasetinsights.io.config_handler import ConfigHandler


@patch("datasetinsights.io.config_handler.create_downloader")
@patch("datasetinsights.io.config_handler.CfgNode.load_cfg")
@patch("datasetinsights.io.config_handler.open")
def test_gcs_load_config(mock_open, mock_CfgNode, mock_create_downloader):
    mock_obj = MagicMock()
    mock_create_downloader.return_value = mock_obj
    mock_open.return_value = mock_obj

    config_url = "gs://test/test.yaml"
    ConfigHandler.load_config(config_url)

    mock_CfgNode.assert_called_with(mock_obj)
    mock_create_downloader.assert_called_with(source_uri=config_url)
    mock_obj.download.assert_called_once()


@patch("datasetinsights.io.config_handler.CfgNode.load_cfg")
@patch("datasetinsights.io.config_handler.open")
def test_local_load_config(mock_open, mock_CfgNode):
    mock_obj = MagicMock()
    mock_open.return_value = mock_obj

    config_url = "test.yaml"
    ConfigHandler.load_config(config_url)
    mock_CfgNode.assert_called_with(mock_obj)
