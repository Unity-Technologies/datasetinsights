from unittest.mock import patch

import pytest
from click import BadParameter
from click.testing import CliRunner
from yacs.config import CfgNode

from datasetinsights.commands.train import OverrideKey, cli


@pytest.mark.parametrize(
    "args",
    [
        [
            "train",
            "--config=tests/configs/faster_rcnn_groceries_real_test.yaml",
            "--train-data=tests/datasets",
        ],
        [
            "train",
            "-c",
            "tests/configs/faster_rcnn_groceries_real_test.yaml",
            "-t",
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
        train_data="tests/datasets", val_data=None
    )


@pytest.mark.parametrize(
    "args",
    [
        [
            "train",
            "--config" "tests/configs/faster_rcnn_groceries_real_test.yaml",
            "--train-data",
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


def test_override_key_validation():
    validate_override_key = OverrideKey()

    override_key1 = "optimizer.args.lr=0.00005 pretrained=False"
    override_key2 = "test_a.b=c"

    assert validate_override_key(override_key1) == override_key1
    assert validate_override_key(override_key2) == override_key2

    with pytest.raises(BadParameter):
        validate_override_key("test_a.b")
        validate_override_key("optimizer.args.lr=0.00005pretrained=False")
        validate_override_key("optimizer.args.lr=0.00005pretrained=False ")
