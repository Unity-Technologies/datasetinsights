import pytest
from yacs.config import CfgNode as CN

from datasetinsights.estimators import FasterRCNN, create_estimator


@pytest.fixture
def config():
    """prepare config."""
    with open("tests/configs/faster_rcnn_groceries_real_test.yaml") as f:
        cfg = CN.load_cfg(f)

    return cfg


def test_create_estimator_with_config_expect_faster_rcnn(config):

    estimator = create_estimator(name=config.estimator, config=config,)

    assert isinstance(estimator, FasterRCNN)
