"""unit test case for frcnn train and evaluate."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
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
    dataloader_creator,
)
from datasetinsights.io.checkpoint import EstimatorCheckpoint

# XXX This should not be a global variable. A tempdir should be a fixture and
# automatically cleanup after EVERY unit test finished execution.
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


def test_faster_rcnn_train_one_epoch(config, dataset):
    """test train one epoch."""
    writer = MagicMock()

    # XXX This is just a hot fix to prevent a mysterious folder such as:
    # <MagicMock name='mock.logdir' id='140420520377936'> showed up after
    # running this test.
    writer.logdir = tmp_name

    kfp_writer = MagicMock()
    checkpointer = MagicMock()
    estimator = FasterRCNN(
        config=config,
        writer=writer,
        checkpointer=checkpointer,
        kfp_writer=kfp_writer,
        logdir="/tmp",
    )
    estimator.writer = writer
    estimator.kfp_writer = kfp_writer
    estimator.checkpointer = checkpointer
    estimator.device = torch.device("cpu")
    train_dataset = dataset
    is_distributed = False
    train_sampler = FasterRCNN.create_sampler(
        is_distributed=is_distributed, dataset=train_dataset, is_train=True
    )
    train_loader = dataloader_creator(
        config, train_dataset, train_sampler, TRAIN, is_distributed
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


@patch("datasetinsights.estimators.faster_rcnn.FasterRCNN.train_one_epoch")
@patch("datasetinsights.estimators.faster_rcnn.Loss.compute")
def test_faster_rcnn_train_all(
    mock_loss, mock_train_one_epoch, config, dataset
):
    """test train on all epochs."""
    loss_val = 0.1
    mock_loss.return_value = loss_val
    log_dir = os.path.join(tmp_name, "train")
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    writer = MagicMock()

    # XXX This is just a hot fix to prevent a mysterious folder such as:
    # <MagicMock name='mock.logdir' id='140420520377936'> showed up after
    # running this test.
    writer.logdir = tmp_name

    kfp_writer = MagicMock()

    checkpointer = EstimatorCheckpoint(
        estimator_name=config.estimator,
        checkpoint_dir=log_dir,
        distributed=False,
    )

    estimator = FasterRCNN(
        config=config,
        writer=writer,
        checkpointer=checkpointer,
        kfp_writer=kfp_writer,
        logdir="/tmp",
    )
    estimator.writer = writer
    estimator.kfp_writer = kfp_writer
    estimator.checkpointer = checkpointer

    estimator.device = torch.device("cpu")
    checkpointer.save = MagicMock()
    train_dataset = dataset
    val_dataset = dataset
    label_mappings = train_dataset.label_mappings
    is_distributed = False
    train_sampler = FasterRCNN.create_sampler(
        is_distributed=is_distributed, dataset=train_dataset, is_train=True
    )
    val_sampler = FasterRCNN.create_sampler(
        is_distributed=is_distributed, dataset=val_dataset, is_train=False
    )

    train_loader = dataloader_creator(
        config, train_dataset, train_sampler, TRAIN, is_distributed
    )
    val_loader = dataloader_creator(
        config, val_dataset, val_sampler, VAL, is_distributed
    )
    epoch = 0
    estimator.train_loop(
        train_dataloader=train_loader,
        label_mappings=label_mappings,
        val_dataloader=val_loader,
        train_sampler=train_sampler,
    )
    writer.add_scalar.assert_called_with("val/loss", loss_val, epoch)
    mock_train_one_epoch.assert_called_once()


@patch("datasetinsights.estimators.faster_rcnn.FasterRCNN.train_loop")
@patch("datasetinsights.estimators.faster_rcnn.Loss.compute")
@patch("datasetinsights.estimators.faster_rcnn.create_dataset")
def test_faster_rcnn_train(
    mock_create, mock_loss, mock_train_loop, config, dataset
):
    """test train."""
    loss_val = 0.1
    mock_loss.return_value = loss_val
    mock_create.return_value = dataset

    kfp_writer = MagicMock()
    writer = MagicMock()

    # XXX This is just a hot fix to prevent a mysterious folder such as:
    # <MagicMock name='mock.logdir' id='140420520377936'> showed up after
    # running this test.
    writer.logdir = tmp_name

    writer.add_scalar = MagicMock()
    writer.add_scalars = MagicMock()
    writer.add_figure = MagicMock()
    checkpointer = MagicMock()
    estimator = FasterRCNN(
        config=config,
        writer=writer,
        checkpointer=checkpointer,
        kfp_writer=kfp_writer,
        logdir="/tmp",
    )
    estimator.checkpointer = checkpointer
    estimator.kfp_writer = kfp_writer
    estimator.writer = writer
    estimator.train(train_data=None)
    mock_train_loop.assert_called_once()


@patch("datasetinsights.estimators.faster_rcnn.Loss.compute")
def test_faster_rcnn_evaluate_per_epoch(mock_loss, config, dataset):
    """test evaluate per epoch."""
    loss_val = 0.1
    mock_loss.return_value = loss_val
    ckpt_dir = tmp_name + "/train/FasterRCNN.estimator"
    config.checkpoint_file = ckpt_dir
    writer = MagicMock()

    # XXX This is just a hot fix to prevent a mysterious folder such as:
    # <MagicMock name='mock.logdir' id='140420520377936'> showed up after
    # running this test.
    writer.logdir = tmp_name

    kfp_writer = MagicMock()
    checkpointer = MagicMock()
    writer.add_scalar = MagicMock()

    estimator = FasterRCNN(
        config=config,
        writer=writer,
        checkpointer=checkpointer,
        kfp_writer=kfp_writer,
        logdir="/tmp",
    )
    estimator.writer = writer
    estimator.kfp_writer = kfp_writer
    estimator.checkpointer = checkpointer

    estimator.device = torch.device("cpu")

    test_dataset = dataset
    label_mappings = test_dataset.label_mappings
    is_distributed = False
    test_sampler = FasterRCNN.create_sampler(
        is_distributed=is_distributed, dataset=test_dataset, is_train=False
    )
    test_loader = dataloader_creator(
        config, test_dataset, test_sampler, TEST, is_distributed
    )
    sync_metrics = config.get("synchronize_metrics", True)
    epoch = 0
    estimator.evaluate_per_epoch(
        data_loader=test_loader,
        epoch=epoch,
        label_mappings=label_mappings,
        synchronize_metrics=sync_metrics,
    )
    writer.add_scalar.assert_called_with("val/loss", loss_val, epoch)


@patch("datasetinsights.estimators.faster_rcnn.FasterRCNN.evaluate_per_epoch")
@patch("datasetinsights.estimators.faster_rcnn.Loss.compute")
@patch("datasetinsights.estimators.faster_rcnn.create_dataset")
def test_faster_rcnn_evaluate(
    mock_create, mock_loss, mock_evaluate_per_epoch, config, dataset
):
    """test evaluate."""
    mock_create.return_value = dataset
    loss_val = 0.1
    mock_loss.return_value = loss_val
    ckpt_dir = tmp_name + "/train/FasterRCNN.estimator"
    config.checkpoint_file = ckpt_dir
    writer = MagicMock()

    # XXX This is just a hot fix to prevent a mysterious folder such as:
    # <MagicMock name='mock.logdir' id='140420520377936'> showed up after
    # running this test.
    writer.logdir = tmp_name

    kfp_writer = MagicMock()
    checkpointer = MagicMock()
    writer.add_scalar = MagicMock()
    estimator = FasterRCNN(
        config=config,
        writer=writer,
        checkpointer=checkpointer,
        kfp_writer=kfp_writer,
        logdir="/tmp",
    )
    estimator.writer = writer
    estimator.kfp_writer = kfp_writer
    estimator.checkpointer = checkpointer
    estimator.evaluate(None)
    mock_evaluate_per_epoch.assert_called_once()


def test_faster_rcnn_log_metric_val(config):
    """test log metric val."""
    writer = MagicMock()

    # XXX This is just a hot fix to prevent a mysterious folder such as:
    # <MagicMock name='mock.logdir' id='140420520377936'> showed up after
    # running this test.
    writer.logdir = tmp_name

    kfp_writer = MagicMock()
    checkpointer = MagicMock()
    writer.add_scalar = MagicMock()
    writer.add_scalars = MagicMock()
    writer.add_figure = MagicMock()
    estimator = FasterRCNN(
        config=config,
        writer=writer,
        checkpointer=checkpointer,
        kfp_writer=kfp_writer,
        logdir="/tmp",
    )
    estimator.writer = writer
    estimator.kfp_writer = kfp_writer
    estimator.checkpointer = checkpointer

    estimator.device = torch.device("cpu")
    epoch = 0
    estimator.log_metric_val({"1": "car", "2": "bike"}, epoch)

    writer.add_scalars.assert_called_with("val/APIOU50-per-class", {}, epoch)


# XXX: test_faster_rcnn_save and test_faster_rcnn_load are not independent
# unittests. If test_faster_rcnn_save is removed, test_faster_rcnn_load will
# fail. They should be completely independent.
def test_faster_rcnn_save(config):
    """test save model."""

    log_dir = tmp_name + "/train/"
    kfp_writer = MagicMock()
    writer = MagicMock()

    # XXX This is just a hot fix to prevent a mysterious folder such as:
    # <MagicMock name='mock.logdir' id='140420520377936'> showed up after
    # running this test.
    writer.logdir = tmp_name

    checkpointer = EstimatorCheckpoint(
        estimator_name=config.estimator,
        checkpoint_dir=log_dir,
        distributed=False,
    )
    estimator = FasterRCNN(
        config=config,
        writer=writer,
        checkpointer=checkpointer,
        kfp_writer=kfp_writer,
        logdir="/tmp",
    )
    estimator.writer = writer
    estimator.kfp_writer = kfp_writer
    estimator.checkpointer = checkpointer
    estimator.device = torch.device("cpu")
    estimator.save(log_dir + "FasterRCNN.estimator")

    assert any(
        [
            name.startswith("FasterRCNN.estimator")
            for name in os.listdir(log_dir)
        ]
    )


def test_faster_rcnn_load(config):
    """test load model."""

    ckpt_dir = tmp_name + "/train/FasterRCNN.estimator"
    config.checkpoint_file = ckpt_dir
    log_dir = tmp_name + "/load/"
    config.logdir = log_dir
    kfp_writer = MagicMock()
    writer = SummaryWriter(config.logdir, write_to_disk=True)
    checkpointer = EstimatorCheckpoint(
        estimator_name=config.estimator,
        checkpoint_dir=log_dir,
        distributed=False,
    )
    estimator = FasterRCNN(
        config=config,
        writer=writer,
        checkpointer=checkpointer,
        kfp_writer=kfp_writer,
        logdir="/tmp",
    )
    estimator.writer = writer
    estimator.kfp_writer = kfp_writer
    estimator.checkpointer = checkpointer

    estimator.device = torch.device("cpu")
    estimator.load(ckpt_dir)
    assert os.listdir(log_dir)[0].startswith("events.out.tfevents")


def test_len_dataset(config, dataset):
    """test download data."""
    assert len(dataset.images) == len(dataset)


@patch("datasetinsights.estimators.faster_rcnn.Dataset.create")
def test_create_sampler(mock_create, config, dataset):
    """test create sampler."""
    mock_create.return_value = dataset
    is_distributed = False
    train_sampler = FasterRCNN.create_sampler(
        is_distributed=is_distributed, dataset=dataset, is_train=True
    )
    assert len(dataset.images) == len(train_sampler)


@patch("datasetinsights.estimators.faster_rcnn.torch.utils.data.DataLoader")
def test_dataloader_creator(mock_loader, config, dataset):
    """test create dataloader."""
    mock_loader.return_value = MagicMock()
    is_distributed = False
    train_sampler = FasterRCNN.create_sampler(
        is_distributed=is_distributed, dataset=dataset, is_train=True
    )
    train_loader = dataloader_creator(
        config, dataset, train_sampler, TRAIN, is_distributed
    )
    assert isinstance(train_loader, MagicMock)


@patch("datasetinsights.estimators.faster_rcnn.torch.utils.data.DataLoader")
def test_create_dataloader(mock_loader, config, dataset):
    """test load data."""

    mock_loader.return_value = MagicMock()
    is_distributed = False
    train_sampler = FasterRCNN.create_sampler(
        is_distributed=is_distributed, dataset=dataset, is_train=True
    )
    dataloader = create_dataloader(
        distributed=is_distributed,
        dataset=dataset,
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
def test_create_optimizer(mock_lr, mock_adm, config, dataset):
    """test create optimizer."""
    mock_lr.return_value = MagicMock()
    mock_adm.return_value = MagicMock()

    writer = MagicMock()

    # XXX This is just a hot fix to prevent a mysterious folder such as:
    # <MagicMock name='mock.logdir' id='140420520377936'> showed up after
    # running this test.
    writer.logdir = tmp_name

    kfp_writer = MagicMock()
    checkpointer = MagicMock()
    estimator = FasterRCNN(
        config=config,
        writer=writer,
        checkpointer=checkpointer,
        kfp_writer=kfp_writer,
        logdir="/tmp",
    )
    estimator.writer = writer
    estimator.kfp_writer = kfp_writer
    estimator.checkpointer = checkpointer

    estimator.device = torch.device("cpu")
    params = [p for p in estimator.model.parameters() if p.requires_grad]
    optimizer, lr_scheduler = FasterRCNN.create_optimizer_lrs(config, params)

    assert isinstance(optimizer, MagicMock)
    assert isinstance(lr_scheduler, MagicMock)


def test_faster_rcnn_predict(config, dataset):
    """test predict."""

    checkpoint_file = tmp_name + "/train/FasterRCNN.estimator"
    kfp_writer = MagicMock()
    writer = MagicMock()

    # XXX This is just a hot fix to prevent a mysterious folder such as:
    # <MagicMock name='mock.logdir' id='140420520377936'> showed up after
    # running this test.
    writer.logdir = tmp_name

    checkpointer = EstimatorCheckpoint(
        estimator_name=config.estimator,
        checkpoint_dir="/tmp",
        distributed=False,
    )
    estimator = FasterRCNN(
        config=config,
        writer=writer,
        checkpointer=checkpointer,
        kfp_writer=kfp_writer,
        checkpoint_file=checkpoint_file,
        logdir="/tmp",
    )
    estimator.writer = writer
    estimator.kfp_writer = kfp_writer
    estimator.checkpointer = checkpointer

    estimator.device = torch.device("cpu")
    image_size = (256, 256)
    image = Image.fromarray(np.random.random(image_size), "L")

    result = estimator.predict(image)
    assert result == []


# XXX This test should be completely removed. A tempdir cleanup should happen
# EVERY unit test is finished execution.
def test_clean_dir():
    """clean tmp dir."""
    if os.path.exists(tmp_dir.name):
        tmp_dir.cleanup()
