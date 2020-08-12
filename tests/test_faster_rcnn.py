"""unit test case for frcnn train and evaluate."""

import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torchvision
from PIL import Image
from tensorboardX import SummaryWriter
from yacs.config import CfgNode as CN

from datasetinsights.datasets.dummy.dummy_object_detection import (
    DummyDetection2D,
)
from datasetinsights.estimators.faster_rcnn import (
    TEST,
    TRAIN,
    VAL,
    FasterRCNN,
    create_dataloader,
    create_dataset,
    create_dryrun_dataset,
    dataloader_creator,
)
from datasetinsights.io.checkpoint import EstimatorCheckpoint

tmp_dir = tempfile.TemporaryDirectory()
tmp_name = tmp_dir.name


@pytest.fixture
def dataset():
    """prepare dataset."""
    dummy_data = DummyDetection2D(transform=FasterRCNN.get_transform())
    return dummy_data


@pytest.fixture
def config():
    """prepare config."""
    with open("tests/configs/faster_rcnn_groceries_real_test.yaml") as f:
        cfg = CN.load_cfg(f)

    return cfg


@patch("datasetinsights.estimators.faster_rcnn.Dataset.create")
def test_faster_rcnn_train_one_epoch(mock_create, config, dataset):
    """test train one epoch."""
    mock_create.return_value = dataset
    writer = MagicMock()
    kfp_writer = MagicMock()
    checkpointer = MagicMock()
    estimator = FasterRCNN(
        config=config,
        writer=writer,
        device=torch.device("cpu"),
        checkpointer=checkpointer,
        kfp_writer=kfp_writer,
    )
    train_dataset = create_dataset(config, TRAIN)
    is_distributed = config.system.distributed
    train_sampler = FasterRCNN.create_sampler(
        is_distributed=is_distributed, dataset=train_dataset, is_train=True
    )
    train_loader = dataloader_creator(
        config, train_dataset, train_sampler, TRAIN
    )
    params = [p for p in estimator.model.parameters() if p.requires_grad]
    optimizer, lr_scheduler = FasterRCNN.create_optimizer_lrs(config, params)
    accumulation_steps = config.train.get("accumulation_steps", 1)
    epoch = 1
    estimator.train_one_epoch(
        optimizer=optimizer,
        data_loader=train_loader,
        epoch=epoch,
        lr_scheduler=lr_scheduler,
        accumulation_steps=accumulation_steps,
    )
    writer.add_scalar.assert_called_with(
        "training/lr", config.optimizer.args.get("lr"), epoch
    )


@patch("datasetinsights.estimators.faster_rcnn.Loss.compute")
@patch("datasetinsights.estimators.faster_rcnn.Dataset.create")
def test_faster_rcnn_train_all(mock_create, mock_loss, config, dataset):
    """test train on all epochs."""
    loss_val = 0.1
    mock_create.return_value = dataset
    mock_loss.return_value = loss_val
    writer = MagicMock()
    kfp_writer = MagicMock()
    checkpointer = MagicMock()
    estimator = FasterRCNN(
        config=config,
        writer=writer,
        device=torch.device("cpu"),
        checkpointer=checkpointer,
        kfp_writer=kfp_writer,
    )
    checkpointer.save = MagicMock()
    train_dataset = create_dataset(config, TRAIN)
    val_dataset = create_dataset(config, VAL)
    label_mappings = train_dataset.label_mappings
    is_distributed = config.system.distributed
    train_sampler = FasterRCNN.create_sampler(
        is_distributed=is_distributed, dataset=train_dataset, is_train=True
    )
    val_sampler = FasterRCNN.create_sampler(
        is_distributed=is_distributed, dataset=val_dataset, is_train=False
    )

    train_loader = dataloader_creator(
        config, train_dataset, train_sampler, TRAIN
    )
    val_loader = dataloader_creator(config, val_dataset, val_sampler, VAL)
    epoch = 0
    estimator.train_loop(
        train_dataloader=train_loader,
        label_mappings=label_mappings,
        val_dataloader=val_loader,
        train_sampler=train_sampler,
    )
    writer.add_scalar.assert_called_with("val/loss", loss_val, epoch)


@patch("datasetinsights.estimators.faster_rcnn.Loss.compute")
@patch("datasetinsights.estimators.faster_rcnn.Dataset.create")
def test_faster_rcnn_train(mock_create, mock_loss, config, dataset):
    """test train."""
    loss_val = 0.1
    mock_loss.return_value = loss_val
    mock_create.return_value = dataset
    log_dir = tmp_name + "/train/"
    config.system.logdir = log_dir
    kfp_writer = MagicMock()
    writer = MagicMock
    writer.add_scalar = MagicMock()
    writer.add_scalars = MagicMock()
    writer.add_figure = MagicMock()

    checkpointer = EstimatorCheckpoint(
        estimator_name=config.estimator,
        log_dir=log_dir,
        distributed=config.system["distributed"],
    )
    estimator = FasterRCNN(
        config=config,
        writer=writer,
        device=torch.device("cpu"),
        checkpointer=checkpointer,
        kfp_writer=kfp_writer,
    )
    estimator.train()
    writer.add_scalar.assert_called_with(
        "val/loss", loss_val, config.train.epochs - 1
    )


@patch("datasetinsights.estimators.faster_rcnn.Loss.compute")
@patch("datasetinsights.estimators.faster_rcnn.Dataset.create")
def test_faster_rcnn_evaluate_per_epoch(
    mock_create, mock_loss, config, dataset
):
    """test evaluate per epoch."""
    loss_val = 0.1
    mock_loss.return_value = loss_val
    mock_create.return_value = dataset
    ckpt_dir = tmp_name + "/train/FasterRCNN.estimator"
    config.checkpoint_file = ckpt_dir
    writer = MagicMock()
    kfp_writer = MagicMock()
    checkpointer = MagicMock()
    writer.add_scalar = MagicMock()
    estimator = FasterRCNN(
        config=config,
        writer=writer,
        device=torch.device("cpu"),
        checkpointer=checkpointer,
        kfp_writer=kfp_writer,
    )
    test_dataset = create_dataset(config, TEST)
    label_mappings = test_dataset.label_mappings
    is_distributed = config.system.distributed
    test_sampler = FasterRCNN.create_sampler(
        is_distributed=is_distributed, dataset=test_dataset, is_train=False
    )
    test_loader = dataloader_creator(config, test_dataset, test_sampler, TEST)
    sync_metrics = config.get("synchronize_metrics", True)
    epoch = 0
    estimator.evaluate_per_epoch(
        data_loader=test_loader,
        epoch=epoch,
        label_mappings=label_mappings,
        is_distributed=config.system.distributed,
        synchronize_metrics=sync_metrics,
    )
    writer.add_scalar.assert_called_with("val/loss", loss_val, epoch)


@patch("datasetinsights.estimators.faster_rcnn.Loss.compute")
@patch("datasetinsights.estimators.faster_rcnn.Dataset.create")
def test_faster_rcnn_evaluate(mock_create, mock_loss, config, dataset):
    """test evaluate."""
    mock_create.return_value = dataset
    loss_val = 0.1
    mock_loss.return_value = loss_val
    ckpt_dir = tmp_name + "/train/FasterRCNN.estimator"
    config.checkpoint_file = ckpt_dir
    writer = MagicMock()
    kfp_writer = MagicMock()
    checkpointer = MagicMock()
    writer.add_scalar = MagicMock()
    estimator = FasterRCNN(
        config=config,
        writer=writer,
        device=torch.device("cpu"),
        checkpointer=checkpointer,
        kfp_writer=kfp_writer,
    )
    estimator.evaluate()
    epoch = 0
    writer.add_scalar.assert_called_with("val/loss", loss_val, epoch)


@patch("datasetinsights.estimators.faster_rcnn.Dataset.create")
def test_faster_rcnn_log_metric_val(mock_create, config, dataset):
    """test log metric val."""
    mock_create.return_value = dataset
    writer = MagicMock()
    kfp_writer = MagicMock()
    checkpointer = MagicMock()
    writer.add_scalar = MagicMock()
    writer.add_scalars = MagicMock()
    writer.add_figure = MagicMock()
    estimator = FasterRCNN(
        config=config,
        writer=writer,
        device=torch.device("cpu"),
        checkpointer=checkpointer,
        kfp_writer=kfp_writer,
    )
    epoch = 0
    estimator.log_metric_val(dataset.label_mappings, epoch)

    writer.add_scalars.assert_called_with("val/AR-per-class", {}, epoch)


@patch("datasetinsights.estimators.faster_rcnn.Dataset.create")
def test_faster_rcnn_save(mock_create, config, dataset):
    """test save model."""
    mock_create.return_value = dataset
    log_dir = tmp_name + "/test_save/"
    config.system.logdir = log_dir
    kfp_writer = MagicMock()
    writer = MagicMock()
    checkpointer = EstimatorCheckpoint(
        estimator_name=config.estimator,
        log_dir=log_dir,
        distributed=config.system["distributed"],
    )
    estimator = FasterRCNN(
        config=config,
        writer=writer,
        device=torch.device("cpu"),
        checkpointer=checkpointer,
        kfp_writer=kfp_writer,
    )
    estimator.save(log_dir + "FasterRCNN_test")

    assert any(
        [name.startswith("FasterRCNN_test") for name in os.listdir(log_dir)]
    )


@patch("datasetinsights.estimators.faster_rcnn.Dataset.create")
def test_faster_rcnn_load(mock_create, config, dataset):
    """test load model."""
    mock_create.return_value = dataset
    ckpt_dir = tmp_name + "/train/FasterRCNN.estimator"
    config.checkpoint_file = ckpt_dir
    log_dir = tmp_name + "/load/"
    config.system.logdir = log_dir
    kfp_writer = MagicMock()
    writer = SummaryWriter(config.system.logdir, write_to_disk=True)
    checkpointer = EstimatorCheckpoint(
        estimator_name=config.estimator,
        log_dir=log_dir,
        distributed=config.system["distributed"],
    )
    estimator = FasterRCNN(
        config=config,
        writer=writer,
        device=torch.device("cpu"),
        checkpointer=checkpointer,
        kfp_writer=kfp_writer,
    )
    estimator.load(ckpt_dir)
    assert os.listdir(log_dir)[0].startswith("events.out.tfevents")


@patch("datasetinsights.estimators.faster_rcnn.Dataset.create")
def test_create_dataset(mock_create, config, dataset):
    """test download data."""
    mock_create.return_value = dataset
    train_dataset = create_dataset(config, TRAIN)
    assert len(dataset.images) == len(train_dataset)


@patch("datasetinsights.estimators.faster_rcnn.Dataset.create")
def test_create_dryrun_dataset(mock_create, config, dataset):
    """test create dryrun dataset."""
    mock_create.return_value = dataset
    train_dataset = create_dataset(config, TRAIN)
    train_dataset = create_dryrun_dataset(config, train_dataset, TRAIN)
    assert config.train.batch_size * 2 == len(train_dataset)


@patch("datasetinsights.estimators.faster_rcnn.Dataset.create")
def test_create_sampler(mock_create, config, dataset):
    """test create sampler."""
    mock_create.return_value = dataset
    train_dataset = create_dataset(config, TRAIN)
    is_distributed = config.system.distributed
    train_sampler = FasterRCNN.create_sampler(
        is_distributed=is_distributed, dataset=train_dataset, is_train=True
    )
    assert len(dataset.images) == len(train_sampler)


@patch("datasetinsights.estimators.faster_rcnn.torch.utils.data.DataLoader")
@patch("datasetinsights.estimators.faster_rcnn.Dataset.create")
def test_dataloader_creator(mock_create, mock_loader, config, dataset):
    """test create dataloader."""
    mock_create.return_value = dataset
    mock_loader.return_value = MagicMock()
    train_dataset = create_dataset(config, TRAIN)
    is_distributed = config.system.distributed
    train_sampler = FasterRCNN.create_sampler(
        is_distributed=is_distributed, dataset=train_dataset, is_train=True
    )
    train_loader = dataloader_creator(
        config, train_dataset, train_sampler, TRAIN
    )
    assert isinstance(train_loader, MagicMock)


@patch("datasetinsights.estimators.faster_rcnn.torch.utils.data.DataLoader")
@patch("datasetinsights.estimators.faster_rcnn.Dataset.create")
def test_create_dataloader(mock_create, mock_loader, config, dataset):
    """test load data."""
    mock_create.return_value = dataset
    mock_loader.return_value = MagicMock()
    train_dataset = create_dataset(config, TRAIN)
    is_distributed = config.system.distributed
    train_sampler = FasterRCNN.create_sampler(
        is_distributed=is_distributed, dataset=train_dataset, is_train=True
    )
    dataloader = create_dataloader(
        config=config,
        dataset=train_dataset,
        batch_size=config.train.batch_size,
        sampler=train_sampler,
        collate_fn=FasterRCNN.collate_fn,
        train=True,
    )
    assert isinstance(dataloader, MagicMock)


@patch("datasetinsights.estimators.faster_rcnn.torch.optim.Adam")
@patch(
    "datasetinsights.estimators.faster_rcnn.torch.optim.lr_scheduler.LambdaLR"
)
@patch("datasetinsights.estimators.faster_rcnn.Dataset.create")
def test_create_optimizer(mock_create, mock_lr, mock_adm, config, dataset):
    """test create optimizer."""
    mock_lr.return_value = MagicMock()
    mock_adm.return_value = MagicMock()

    mock_create.return_value = dataset
    writer = MagicMock()
    kfp_writer = MagicMock()
    checkpointer = MagicMock()
    estimator = FasterRCNN(
        config=config,
        writer=writer,
        device=torch.device("cpu"),
        checkpointer=checkpointer,
        kfp_writer=kfp_writer,
    )
    params = [p for p in estimator.model.parameters() if p.requires_grad]
    optimizer, lr_scheduler = FasterRCNN.create_optimizer_lrs(config, params)

    assert isinstance(optimizer, MagicMock)
    assert isinstance(lr_scheduler, MagicMock)


@patch("datasetinsights.estimators.faster_rcnn.Dataset.create")
def test_faster_rcnn_predict(mock_create, config, dataset):
    """test predict."""
    mock_create.return_value = dataset
    ckpt_dir = tmp_name + "/train/FasterRCNN.estimator"

    config.checkpoint_file = ckpt_dir
    kfp_writer = MagicMock()
    writer = MagicMock()
    checkpointer = EstimatorCheckpoint(
        estimator_name=config.estimator,
        log_dir=config.system.logdir,
        distributed=config.system["distributed"],
    )
    estimator = FasterRCNN(
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


def test_clean_dir():
    """clean tmp dir."""
    shutil.rmtree(tmp_dir.name, ignore_errors=True)
