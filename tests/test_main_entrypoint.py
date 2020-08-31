from unittest.mock import patch

import pytest
from click.testing import CliRunner

from datasetinsights.__main__ import entrypoint


@pytest.mark.parametrize("args", [[], ["-v"], ["-v", "invalid_command"]])
@patch("datasetinsights.__main__.logging")
def test_entrypoint_except_not_called(logger_mock, args):
    # arrange
    runner = CliRunner()
    # act
    runner.invoke(entrypoint, args)
    # assert
    logger_mock.getLogger.assert_not_called()
    logger_mock.getLogger.return_value.setLevel.assert_not_called()


@pytest.mark.parametrize(
    "args", [["-v", "train"], ["-v", "evaluate"], ["-v", "download"]]
)
@patch("datasetinsights.__main__.logging")
def test_entrypoint_except_called_once(logger_mock, args):
    # arrange
    runner = CliRunner()
    # act
    runner.invoke(entrypoint, args)
    # assert
    logger_mock.getLogger.assert_called_once()
    logger_mock.getLogger.return_value.setLevel.assert_called_once()
