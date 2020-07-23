import argparse
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torchvision
from PIL import Image
from tensorboardX import SummaryWriter
from yacs.config import CfgNode as CN

from datasetinsights.data.dummy.dummy_object_detection import DummyDetection2D
from datasetinsights.estimators import Estimator
from datasetinsights.estimators.faster_rcnn import (
    collate_fn,
    create_dataloader,
    create_dryrun_dataset,
    create_optimizer_lrs,
    create_sampler,
    download_data,
    get_transform,
    load_data,
)
from datasetinsights.storage.checkpoint import create_checkpointer

TRAIN = "train"
VAL = "val"
TEST = "test"

label_mappings = {"1": "car", "2": "bike"}
dummy_data = DummyDetection2D(
    transform=get_transform(), label_mappings=label_mappings
)


@pytest.fixture
def config():
    parser = argparse.ArgumentParser(
        description="Datasetinsights Modeling Interface"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="config file for this model",
    )
    cmd_args = parser.parse_args(
        ["--config=tests/configs/faster_rcnn_groceries_real_test.yaml"]
    )
    cfg = CN.load_cfg(open(cmd_args.config, "r"))

    return cfg


@patch("datasetinsights.estimators.faster_rcnn.Dataset.create")
def test_faster_rcnn_train_one_epoch(mock_create, config):
    mock_create.return_value = dummy_data
    config.system.logdir = "/tmp/train_one/"
    kfp_writer = MagicMock()
    writer = SummaryWriter(config.system.logdir, write_to_disk=True)
    checkpointer = create_checkpointer(logdir=writer.logdir, config=config)
    estimator = Estimator.create(
        config.estimator,
        config=config,
        writer=writer,
        device=torch.device("cpu"),
        checkpointer=checkpointer,
        kfp_writer=kfp_writer,
    )

    train_dataset = download_data(config, TRAIN)
    is_distributed = config.system.distributed
    train_sampler = create_sampler(
        is_distributed=is_distributed, dataset=train_dataset, is_train=True
    )
    train_loader = create_dataloader(
        config, train_dataset, train_sampler, TRAIN
    )

    optimizer, lr_scheduler = create_optimizer_lrs(config, estimator.model)
    accumulation_steps = config.train.get("accumulation_steps", 1)
    estimator.train_one_epoch(
        optimizer=optimizer,
        data_loader=train_loader,
        epoch=0,
        lr_scheduler=lr_scheduler,
        accumulation_steps=accumulation_steps,
    )

    assert os.listdir("/tmp/train_one/")[0].startswith("events.out.tfevents")


@patch("datasetinsights.estimators.faster_rcnn.Dataset.create")
def test_faster_rcnn_train_all(mock_create, config):
    mock_create.return_value = dummy_data
    config.system.logdir = "/tmp/train_all/"

    kfp_writer = MagicMock()
    writer = SummaryWriter(config.system.logdir, write_to_disk=True)
    checkpointer = create_checkpointer(logdir=writer.logdir, config=config)
    estimator = Estimator.create(
        config.estimator,
        config=config,
        writer=writer,
        device=torch.device("cpu"),
        checkpointer=checkpointer,
        kfp_writer=kfp_writer,
    )

    train_dataset = download_data(config, TRAIN)
    val_dataset = download_data(config, VAL)
    label_mappings = train_dataset.label_mappings
    is_distributed = config.system.distributed
    train_sampler = create_sampler(
        is_distributed=is_distributed, dataset=train_dataset, is_train=True
    )
    val_sampler = create_sampler(
        is_distributed=is_distributed, dataset=val_dataset, is_train=False
    )

    train_loader = create_dataloader(
        config, train_dataset, train_sampler, TRAIN
    )
    val_loader = create_dataloader(config, val_dataset, val_sampler, VAL)

    estimator.train_loop(
        train_dataloader=train_loader,
        label_mappings=label_mappings,
        val_dataloader=val_loader,
        train_sampler=train_sampler,
    )
    actual_result = len(next(os.walk("/tmp/train_all/val/AP-per-class/"))[1])
    assert actual_result == len(label_mappings)


@patch("datasetinsights.estimators.faster_rcnn.Dataset.create")
def test_faster_rcnn_train(mock_create, config):
    mock_create.return_value = dummy_data
    config.system.logdir = "/tmp/train/"
    kfp_writer = MagicMock()
    writer = SummaryWriter(config.system.logdir, write_to_disk=True)
    checkpointer = create_checkpointer(logdir=writer.logdir, config=config)
    estimator = Estimator.create(
        config.estimator,
        config=config,
        writer=writer,
        device=torch.device("cpu"),
        checkpointer=checkpointer,
        kfp_writer=kfp_writer,
    )
    estimator.train()
    actual_result = len(next(os.walk("/tmp/train/val/AP-per-class/"))[1])
    assert actual_result == len(label_mappings)


@patch("datasetinsights.estimators.faster_rcnn.Dataset.create")
def test_faster_rcnn_evaluate_per_epoch(mock_create, config):
    mock_create.return_value = dummy_data
    config.checkpoint_file = "/tmp/train/FasterRCNN.estimator"
    config.system.logdir = "/tmp/eval_one_ep/"
    kfp_writer = MagicMock()
    writer = SummaryWriter(config.system.logdir, write_to_disk=True)
    checkpointer = create_checkpointer(logdir=writer.logdir, config=config)
    estimator = Estimator.create(
        config.estimator,
        config=config,
        writer=writer,
        device=torch.device("cpu"),
        checkpointer=checkpointer,
        kfp_writer=kfp_writer,
    )
    test_dataset = download_data(config, TEST)
    label_mappings = test_dataset.label_mappings
    is_distributed = config.system.distributed
    test_sampler = create_sampler(
        is_distributed=is_distributed, dataset=test_dataset, is_train=False
    )
    test_loader = create_dataloader(config, test_dataset, test_sampler, TEST)
    sync_metrics = config.get("synchronize_metrics", True)

    estimator.evaluate_per_epoch(
        data_loader=test_loader,
        epoch=0,
        label_mappings=label_mappings,
        is_distributed=config.system.distributed,
        synchronize_metrics=sync_metrics,
    )
    actual_result = len(next(os.walk("/tmp/eval_one_ep/val/AP-per-class/"))[1])
    assert actual_result == len(label_mappings)


@patch("datasetinsights.estimators.faster_rcnn.Dataset.create")
def test_faster_rcnn_evaluate(mock_create, config):
    mock_create.return_value = dummy_data
    config.checkpoint_file = "/tmp/train/FasterRCNN.estimator"
    config.system.logdir = "/tmp/evaluate/"
    kfp_writer = MagicMock()
    writer = SummaryWriter(config.system.logdir, write_to_disk=True)
    checkpointer = create_checkpointer(logdir=writer.logdir, config=config)
    estimator = Estimator.create(
        config.estimator,
        config=config,
        writer=writer,
        device=torch.device("cpu"),
        checkpointer=checkpointer,
        kfp_writer=kfp_writer,
    )
    estimator.evaluate()
    actual_result = len(next(os.walk("/tmp/evaluate/val/AP-per-class/"))[1])
    assert actual_result == len(label_mappings)


@patch("datasetinsights.estimators.faster_rcnn.Dataset.create")
def test_faster_rcnn_log_metric_val(mock_create, config):
    mock_create.return_value = dummy_data
    config.system.logdir = "/tmp/metric_val/"
    kfp_writer = MagicMock()
    writer = SummaryWriter(config.system.logdir, write_to_disk=True)
    checkpointer = create_checkpointer(logdir=writer.logdir, config=config)
    estimator = Estimator.create(
        config.estimator,
        config=config,
        writer=writer,
        device=torch.device("cpu"),
        checkpointer=checkpointer,
        kfp_writer=kfp_writer,
    )
    estimator.log_metric_val(label_mappings, 0)

    assert os.listdir("/tmp/metric_val/")[0].startswith("events.out.tfevents")


@patch("datasetinsights.estimators.faster_rcnn.Dataset.create")
def test_faster_rcnn_save(mock_create, config):
    mock_create.return_value = dummy_data
    config.system.logdir = "/tmp/test_save/"
    kfp_writer = MagicMock()
    writer = SummaryWriter(config.system.logdir, write_to_disk=True)
    checkpointer = create_checkpointer(logdir=writer.logdir, config=config)
    estimator = Estimator.create(
        config.estimator,
        config=config,
        writer=writer,
        device=torch.device("cpu"),
        checkpointer=checkpointer,
        kfp_writer=kfp_writer,
    )
    estimator.save("/tmp/test_save/FasterRCNN_test")

    assert any(
        [
            name.startswith("FasterRCNN_test")
            for name in os.listdir("/tmp/test_save/")
        ]
    )


@patch("datasetinsights.estimators.faster_rcnn.Dataset.create")
def test_faster_rcnn_load(mock_create, config):
    mock_create.return_value = dummy_data
    config.checkpoint_file = "/tmp/train/FasterRCNN.estimator"
    config.system.logdir = "/tmp/load/"
    kfp_writer = MagicMock()
    writer = SummaryWriter(config.system.logdir, write_to_disk=True)
    checkpointer = create_checkpointer(logdir=writer.logdir, config=config)
    estimator = Estimator.create(
        config.estimator,
        config=config,
        writer=writer,
        device=torch.device("cpu"),
        checkpointer=checkpointer,
        kfp_writer=kfp_writer,
    )
    estimator.load("/tmp/train/FasterRCNN.estimator")

    assert os.listdir("/tmp/load/")[0].startswith("events.out.tfevents")


@patch("datasetinsights.estimators.faster_rcnn.Dataset.create")
def test_download_data(mock_create, config):
    mock_create.return_value = dummy_data
    train_dataset = download_data(config, TRAIN)
    assert len(dummy_data.images) == len(train_dataset)


@patch("datasetinsights.estimators.faster_rcnn.Dataset.create")
def test_create_dryrun_dataset(mock_create, config):
    mock_create.return_value = dummy_data
    train_dataset = download_data(config, TRAIN)
    train_dataset = create_dryrun_dataset(config, train_dataset, TRAIN)
    assert config.train.batch_size * 2 == len(train_dataset)


@patch("datasetinsights.estimators.faster_rcnn.Dataset.create")
def test_create_sampler(mock_create, config):
    mock_create.return_value = dummy_data
    train_dataset = download_data(config, TRAIN)
    is_distributed = config.system.distributed
    train_sampler = create_sampler(
        is_distributed=is_distributed, dataset=train_dataset, is_train=True
    )
    assert len(dummy_data.images) == len(train_sampler)


@patch("datasetinsights.estimators.faster_rcnn.Dataset.create")
def test_create_dataloader(mock_create, config):
    mock_create.return_value = dummy_data
    train_dataset = download_data(config, TRAIN)
    is_distributed = config.system.distributed
    train_sampler = create_sampler(
        is_distributed=is_distributed, dataset=train_dataset, is_train=True
    )
    train_loader = create_dataloader(
        config, train_dataset, train_sampler, TRAIN
    )
    assert train_loader


@patch("datasetinsights.estimators.faster_rcnn.Dataset.create")
def test_load_data(mock_create, config):
    mock_create.return_value = dummy_data
    train_dataset = download_data(config, TRAIN)
    is_distributed = config.system.distributed
    train_sampler = create_sampler(
        is_distributed=is_distributed, dataset=train_dataset, is_train=True
    )
    dataloader = load_data(
        config=config,
        dataset=train_dataset,
        batch_size=config.train.batch_size,
        sampler=train_sampler,
        collate_fn=collate_fn,
        train=True,
    )
    assert dataloader


@patch("datasetinsights.estimators.faster_rcnn.Dataset.create")
def test_create_optimizer(mock_create, config):
    mock_create.return_value = dummy_data
    config.system.logdir = "/tmp/train/"
    kfp_writer = MagicMock()
    writer = SummaryWriter(config.system.logdir, write_to_disk=True)
    checkpointer = create_checkpointer(logdir=writer.logdir, config=config)
    estimator = Estimator.create(
        config.estimator,
        config=config,
        writer=writer,
        device=torch.device("cpu"),
        checkpointer=checkpointer,
        kfp_writer=kfp_writer,
    )
    optimizer, lr_scheduler = create_optimizer_lrs(config, estimator.model)

    assert optimizer
    assert lr_scheduler


@patch("datasetinsights.estimators.faster_rcnn.Dataset.create")
def test_faster_rcnn_predict(mock_create, config):
    mock_create.return_value = dummy_data
    config.checkpoint_file = "/tmp/train/FasterRCNN.estimator"
    config.system.logdir = "/tmp/evaluate/"
    kfp_writer = MagicMock()
    writer = SummaryWriter(config.system.logdir, write_to_disk=True)
    checkpointer = create_checkpointer(logdir=writer.logdir, config=config)
    estimator = Estimator.create(
        config.estimator,
        config=config,
        writer=writer,
        device=torch.device("cpu"),
        checkpointer=checkpointer,
        kfp_writer=kfp_writer,
    )
    image_size = (256, 256)
    image = Image.fromarray(np.random.random(image_size), "L")
    image = torchvision.transforms.functional.to_tensor(image)
    result = estimator.predict(image)
    assert result == []
