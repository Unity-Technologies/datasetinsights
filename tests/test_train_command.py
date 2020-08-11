from unittest.mock import patch

import pytest
from click.testing import CliRunner
from yacs.config import CfgNode

from datasetinsights.commands.train import cli


@pytest.mark.parametrize(
    "args",
    [
        [
            "train",
            "--config=tests/configs/faster_rcnn_groceries_real_test.yaml",
            "--data-root=tests/datasets",
        ],
        [
            "train",
            "-c",
            "tests/configs/faster_rcnn_groceries_real_test.yaml",
            "-d",
            "tests/datasets",
        ],
    ],
)
@patch("builtins.open")
@patch.object(CfgNode, "load_cfg")
@patch("datasetinsights.commands.train.create_estimator")
def test_train_except_called_once(
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
    estimator_factory_create_mock.return_value.train.assert_called_once_with(
        data_root="tests/datasets"
    )


@pytest.mark.parametrize(
    "args",
    [
        [
            "train",
            "--config" "tests/configs/faster_rcnn_groceries_real_test.yaml",
            "--data-root",
            "invalid-data-root",
        ],
        ["train"],
    ],
)
@patch("builtins.open")
@patch.object(CfgNode, "load_cfg")
@patch("datasetinsights.estimators.base.create_estimator")
def test_train_except_not_called(
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
    estimator_factory_create_mock.return_value.train.assert_not_called()
