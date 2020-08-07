from unittest.mock import patch

import pytest
from click.testing import CliRunner
from yacs.config import CfgNode

from datasetinsights.commands.evaluate import cli
from datasetinsights.estimators.factory import EstimatorFactory


@pytest.mark.parametrize(
    "args",
    [
        [
            "evaluate",
            "--config=tests/configs/faster_rcnn_groceries_real_test.yaml",
            "--checkpoint-file=checkpoint_file.txt",
            f"--data-root=tests/datasets",
        ],
        [
            "evaluate",
            "-c",
            "tests/configs/faster_rcnn_groceries_real_test.yaml",
            "-d",
            "tests/datasets",
            "-p",
            "--checkpoint=checkpoint_file.txt",
        ],
    ],
)
@patch("builtins.open")
@patch.object(CfgNode, "load_cfg")
@patch.object(EstimatorFactory, "create")
def test_evaluate_except_called_once(
    estimator_factory_create_mock, cfg_node_mock, open_mock, args
):
    # arrange
    runner = CliRunner()
    # act

    runner.invoke(cli, args)
    # assert
    open_mock.assert_called_once_with(
        "tests/configs/faster_rcnn_groceries_real_test.yaml", "r"
    )
    cfg_node_mock.assert_called_once()
    estimator_factory_create_mock.assert_called_once()
    estimator_factory_create_mock.return_value.evaluate.assert_called_once()


@pytest.mark.parametrize(
    "args",
    [
        ["evaluate"],
        [
            "evaluate",
            "--config",
            "tests/configs/faster_rcnn_groceries_real_test.yaml",
            "--data-root",
            "invalid-data-root",
        ],
    ],
)
@patch("builtins.open")
@patch.object(CfgNode, "load_cfg")
@patch.object(EstimatorFactory, "create")
def test_evaluate_except_not_called(
    estimator_factory_create_mock, cfg_node_mock, open_mock, args
):
    # arrange
    runner = CliRunner()
    # act
    runner.invoke(cli, args)
    # assert
    open_mock.assert_not_called()
    cfg_node_mock.assert_not_called()
    estimator_factory_create_mock.assert_not_called()
    estimator_factory_create_mock.return_value.evaluate.assert_not_called()
