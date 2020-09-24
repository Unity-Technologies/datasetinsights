"""faster rcnn pytorch train and evaluate."""

import copy
import logging
import math
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torchvision
from codetiming import Timer
from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

import datasetinsights.constants as const
from datasetinsights.datasets import Dataset
from datasetinsights.evaluation_metrics.base import EvaluationMetric
from datasetinsights.io.bbox import BBox2D
from datasetinsights.io.transforms import Compose
from datasetinsights.torch_distributed import get_world_size, is_master

from .base import Estimator

MAX_BOXES_PER_IMAGE = 100
logger = logging.getLogger(__name__)
DEFAULT_ACCUMULATION_STEPS = 1
TRAIN = "train"
VAL = "val"
TEST = "test"


class FasterRCNN(Estimator):
    """
    Faster-RCNN train/evaluate implementation for object detection.

    https://github.com/pytorch/vision/tree/master/references/detection
    https://arxiv.org/abs/1506.01497
    Args:
        config (CfgNode): estimator config
        box_score_thresh: (optional) default threshold is 0.05
        distributed: whether or not the estimator is distributed
        kfp_metrics_filename: Kubeflow Metrics filename
        kfp_metrics_dir: Path to the directory where Kubeflow
         metrics files are stored


    https://github.com/pytorch/vision/tree/master/references/detection
    https://arxiv.org/abs/1506.01497

    Attributes:
        model: pytorch model
        writer: Tensorboard writer object
        kfp_writer: KubeflowPipelineWriter object
        checkpointer: Model checkpointer callback to save models
        device: model training on device (cpu|cuda)
    """

    def __init__(
        self,
        *,
        config,
        writer,
        kfp_writer,
        checkpointer,
        box_score_thresh=0.05,
        no_cuda=None,
        checkpoint_file=None,
        **kwargs,
    ):
        """initiate estimator."""

        logger.info(f"initializing faster rcnn")
        self.config = config

        self._init_distributed_mode()
        self.no_cuda = no_cuda
        self._init_device()
        self.writer = SummaryWriter(writer.logdir, write_to_disk=is_master())

        self.kfp_writer = kfp_writer
        checkpointer.distributed = self.distributed
        self.checkpointer = checkpointer

        model_name = f"fasterrcnn_{self.config.backbone}_fpn"
        self.model = torchvision.models.detection.__dict__[model_name](
            num_classes=config.num_classes,
            pretrained_backbone=config.pretrained_backbone,
            pretrained=config.pretrained,
            box_detections_per_img=MAX_BOXES_PER_IMAGE,
            box_score_thresh=box_score_thresh,
        )
        self.model_without_ddp = self.model
        self.sync_metrics = config.get("synchronize_metrics", True)
        self.metrics = {}
        for metric_key, metric in config.metrics.items():
            self.metrics[metric_key] = EvaluationMetric.create(metric.name)
        self.model.to(self.device)

        if self.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.gpu]
            )
            self.model_without_ddp = self.model.module

        if checkpoint_file:
            self.checkpointer.load(self, checkpoint_file)

    def _init_distributed_mode(self):
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            logger.info(f"found RANK and WORLD_SIZE in environment")
            self.rank = int(os.environ["RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.gpu = int(os.environ["LOCAL_RANK"])
        elif "SLURM_PROCID" in os.environ:
            logger.info(f"found 'SLURM_PROCID' in environment")
            self.rank = int(os.environ["SLURM_PROCID"])
            self.gpu = self.rank % torch.cuda.device_count()
        else:
            self.gpu = 0
            self.rank = 0
            logger.info("Not using distributed mode")
            self.distributed = False
            return

        device_count = torch.cuda.device_count()
        logger.info(f"device count: {torch.cuda.device_count()}")
        logger.info(f"world size: {self.world_size}")
        logger.info(f"gpu: {self.gpu}")
        logger.info(f"local rank {self.rank}")
        if device_count == 0:
            logger.info("No cuda devices found, will not parallelize")
            self.distributed = False
            return
        if not is_master():
            logging.disable(logging.ERROR)
        self.distributed = True
        torch.cuda.set_device(self.gpu)

        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=self.world_size,
            rank=self.rank,
        )
        torch.distributed.barrier()

    def _init_device(self):
        if torch.cuda.is_available() and not self.no_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def train(self, train_data, val_data=None, **kwargs):
        """start training, save trained model per epoch.

        Args:
            train_data: Directory on localhost where train dataset is located.
            val_data: Directory on localhost where
            validation dataset is located.

        """
        config = self.config
        train_dataset = create_dataset(config, train_data, TRAIN)
        val_dataset = (
            create_dataset(config, val_data, VAL) if val_data else None
        )

        label_mappings = train_dataset.label_mappings

        logger.info(f"length of train dataset is {len(train_dataset)}")
        if val_dataset:
            logger.info(f"length of validation dataset is {len(val_dataset)}")

        train_sampler = FasterRCNN.create_sampler(
            is_distributed=self.distributed,
            dataset=train_dataset,
            is_train=True,
        )
        val_sampler = FasterRCNN.create_sampler(
            is_distributed=self.distributed, dataset=val_dataset, is_train=False
        )

        train_loader = dataloader_creator(
            config, train_dataset, train_sampler, TRAIN, self.distributed
        )
        val_loader = dataloader_creator(
            config, val_dataset, val_sampler, VAL, self.distributed
        )
        self.train_loop(
            train_dataloader=train_loader,
            label_mappings=label_mappings,
            val_dataloader=val_loader,
            train_sampler=train_sampler,
        )
        self.writer.close()
        self.kfp_writer.write_metric()

    def train_loop(
        self,
        *,
        train_dataloader,
        label_mappings,
        val_dataloader,
        train_sampler=None,
    ):
        """train on whole range of epochs.

        Args:
            train_dataloader (torch.utils.data.DataLoader):
            label_mappings (dict): a dict of {label_id: label_name} mapping
            val_dataloader (torch.utils.data.DataLoader)
            train_sampler: (torch.utils.data.Sampler)
        """
        model = self.model.to(self.device)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer, lr_scheduler = FasterRCNN.create_optimizer_lrs(
            self.config, params
        )
        accumulation_steps = self.config.train.get(
            "accumulation_steps", DEFAULT_ACCUMULATION_STEPS
        )
        logger.debug("Start training")
        total_timer = Timer(
            name="total-time", text=const.TIMING_TEXT, logger=logging.info
        )
        total_timer.start()
        for epoch in range(self.config.train.epochs):
            with Timer(
                name=f"epoch-{epoch}-train-time",
                text=const.TIMING_TEXT,
                logger=logging.info,
            ):
                self.train_one_epoch(
                    optimizer=optimizer,
                    data_loader=train_dataloader,
                    epoch=epoch,
                    lr_scheduler=lr_scheduler,
                    accumulation_steps=accumulation_steps,
                )
            if self.distributed:
                train_sampler.set_epoch(epoch)
            self.checkpointer.save(self, epoch=epoch)
            with Timer(
                name=f"epoch-{epoch}-evaluate-time",
                text=const.TIMING_TEXT,
                logger=logging.info,
            ):
                self.evaluate_per_epoch(
                    data_loader=val_dataloader,
                    epoch=epoch,
                    label_mappings=label_mappings,
                )
        total_timer.stop()

    def train_one_epoch(
        self,
        *,
        optimizer,
        data_loader,
        epoch,
        lr_scheduler,
        accumulation_steps,
    ):
        """train per epoch.

        Args:
            optimizer: pytorch optimizer
            data_loader(DataLoader): pytorch dataloader
            epoch (int): lr_scheduler: Pytorch LR scheduler
            lr_scheduler: Pytorch LR scheduler
            accumulation_steps(int): Accumulated Gradients are only updated
            after X steps. This creates an effective batch size of
            batch_size * accumulation_steps
        """
        self.model.train()
        loss_metric = Loss()
        optimizer.zero_grad()
        n_examples = len(data_loader.dataset)
        for i, (images, targets) in enumerate(data_loader):
            images = list(image.to(self.device) for image in images)
            targets = [
                {k: v.to(self.device) for k, v in t.items()} for t in targets
            ]
            loss_dict = self.model(images, targets)

            losses_grad = sum(loss for loss in loss_dict.values())

            loss_dict_reduced = reduce_dict(loss_dict)
            losses = sum(loss for loss in loss_dict_reduced.values())
            loss_metric.update(avg_loss=losses.item(), batch_size=len(targets))

            if not math.isfinite(losses):
                raise BadLoss(
                    f"Loss is {losses}, stopping training. Input was "
                    f"{images, targets}. Loss dict is {loss_dict}"
                )
            elif i % self.config.train.log_frequency == 0:
                intermediate_loss = loss_metric.compute()
                examples_seen = epoch * n_examples + loss_metric.num_examples
                logger.info(
                    f"intermediate loss after mini batch {i} in epoch {epoch} "
                    f"(total training examples: {examples_seen}) is "
                    f"{intermediate_loss}"
                )
                self.writer.add_scalar(
                    "training/intermediate_loss",
                    intermediate_loss,
                    examples_seen,
                )
            losses_grad.backward()
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
        self.writer.add_scalar("training/loss", loss_metric.compute(), epoch)
        self.writer.add_scalar(
            "training/running_loss",
            loss_metric.compute(),
            loss_metric.num_examples,
        )
        self.writer.add_scalar(
            "training/lr", optimizer.param_groups[0]["lr"], epoch
        )
        loss_metric.reset()

    def evaluate(self, test_data, **kwargs):
        """evaluate given dataset."""
        config = self.config
        test_dataset = create_dataset(config, test_data, TEST)
        label_mappings = test_dataset.label_mappings
        test_sampler = FasterRCNN.create_sampler(
            is_distributed=self.distributed,
            dataset=test_dataset,
            is_train=False,
        )

        logger.info(f"length of test dataset is {len(test_dataset)}")
        logger.info("Start evaluating estimator: %s", type(self).__name__)

        test_loader = dataloader_creator(
            config, test_dataset, test_sampler, TEST, self.distributed
        )
        self.model.to(self.device)
        self.evaluate_per_epoch(
            data_loader=test_loader,
            epoch=0,
            label_mappings=label_mappings,
            synchronize_metrics=self.sync_metrics,
        )
        self.writer.close()
        self.kfp_writer.write_metric()

    @torch.no_grad()
    def evaluate_per_epoch(
        self,
        *,
        data_loader,
        epoch,
        label_mappings,
        max_detections_per_img=MAX_BOXES_PER_IMAGE,
        synchronize_metrics=True,
    ):
        """Evaluate model performance per epoch.

        Note, torchvision's implementation of faster
        rcnn requires input and gt data for training mode and returns a
        dictionary of losses (which we need to record the loss). We also need to
        get the raw predictions, which is only possible in model.eval() mode,
        to calculate the evaluation metric.
        Args:
            data_loader (DataLoader): pytorch dataloader
            epoch (int): current epoch, used for logging
            label_mappings (dict): a dict of {label_id: label_name} mapping
            max_detections_per_img: max number of targets or predictions allowed
            per example
            is_distributed: whether or not the model is distributed
            synchronize_metrics: whether or not to synchronize evaluation
            metrics across processes
        Returns:

        """
        n_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.model.eval()

        loss_metric = Loss()
        for metric in self.metrics.values():
            metric.reset()
        for i, (images, targets) in enumerate(tqdm(data_loader)):
            images = list(image.to(self.device) for image in images)
            targets_raw = [
                {k: v.to(self.device) for k, v in t.items()} for t in targets
            ]
            targets_converted = convert_bboxes2canonical(targets)
            # we need 2 do forward passes, one in eval mode gets us the raw
            # preds and one in train mode gets us the losses
            # pass number 1 to get mAP
            self.model.eval()
            preds_raw = self.model(images)
            converted_preds = convert_bboxes2canonical(preds_raw)
            gt_preds = list(zip(targets_converted, converted_preds))
            if self.distributed and synchronize_metrics:
                all_preds_gt_canonical = gather_gt_preds(
                    gt_preds=gt_preds,
                    device=self.device,
                    max_boxes=max_detections_per_img,
                )
            else:
                all_preds_gt_canonical = gt_preds
            for metric in self.metrics.values():
                metric.update(all_preds_gt_canonical)

            # pass number 2 to get val loss
            self.model.train()
            loss_dict = self.model(images, targets_raw)
            loss_dict_reduced = reduce_dict(loss_dict)
            losses = sum(loss for loss in loss_dict_reduced.values())
            loss_metric.update(avg_loss=losses.item(), batch_size=len(targets))

        self.log_metric_val(label_mappings, epoch)
        val_loss = loss_metric.compute()
        logger.info(f"validation loss is {val_loss}")
        self.writer.add_scalar("val/loss", val_loss, epoch)

        torch.set_num_threads(n_threads)

    def log_metric_val(self, label_mappings, epoch):
        """log metric values.

        Args:
            label_mappings (dict): a dict of {label_id: label_name} mapping
            epoch (int): current epoch, used for logging
        """
        for metric_name, metric in self.metrics.items():
            result = metric.compute()
            logger.debug(result)
            logger.info(f"metric {metric_name} has result: {result}")
            if metric.TYPE == "scalar":
                self.writer.add_scalar(f"val/{metric_name}", result, epoch)
                self.kfp_writer.add_metric(name=metric_name, val=result)
            # TODO (YC) This is hotfix to allow user map between label_id
            # to label_name during model evaluation. In ideal cases this mapping
            # should be available before training/evaluation dataset was loaded.
            # label_id that was missing from label_name should be removed from
            # dataset and the training procedure.
            elif metric.TYPE == "metric_per_label":
                label_results = {
                    label_mappings.get(id, str(id)): value
                    for id, value in result.items()
                }
                self.writer.add_scalars(
                    f"val/{metric_name}-per-class", label_results, epoch
                )
                fig = metric_per_class_plot(metric_name, result, label_mappings)
                self.writer.add_figure(f"{metric_name}-per-class", fig, epoch)

    def save(self, path):
        """Serialize Estimator to path.

        Args:
            path (str): full path to save serialized estimator

        Returns:
            saved full path of the serialized estimator
        """
        save_dict = {
            "model": self.model_without_ddp.state_dict(),
            "config": self.config,
        }
        torch.save(save_dict, path)

        return path

    def load(self, path):
        """Load Estimator from path.

        Args:
            path (str): full path to the serialized estimator
        """
        logger.info(f"loading checkpoint from file")
        checkpoint = torch.load(path, map_location=self.device)
        self.model_without_ddp.load_state_dict(checkpoint["model"])
        loaded_config = copy.deepcopy(checkpoint["config"])
        stored_config = copy.deepcopy(self.config)
        if stored_config != loaded_config:
            logger.debug(
                f"Found difference in estimator config."
                f"Estimator loaded from {path} was trained using config: \n"
                f"{loaded_config}. \nHowever, the current config is: \n"
                f"{self.config}."
            )
        self.model_without_ddp.eval()

    def predict(self, pil_img, box_score_thresh=0.5):
        """Get prediction from one image using loaded model.

        Args:
            pil_img (PIL Image): PIL image from dataset.
            box_score_thresh (float): box score threshold for filter out lower
            score bounding boxes. Defaults to 0.5.

        Returns:
            filtered_pred_annotations (List[BBox2D]): high predicted score
            bboxes from the model.
        """
        img_tensor = transforms.ToTensor()(pil_img).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)
        predicts = self.model_without_ddp(img_tensor)
        predict_annotations = convert_bboxes2canonical(predicts)[0]
        filtered_pred_annotations = [
            box for box in predict_annotations if box.score >= box_score_thresh
        ]
        return filtered_pred_annotations

    @staticmethod
    def create_sampler(is_distributed, *, dataset, is_train):
        """create sample of data.

        Args:
            is_distributed: whether or not the model is distributed
            dataset: dataset obj must have len and __get_item__
            is_train: whether or not the sampler is for training data

        Returns:
            data_sampler: (torch.utils.data.Sampler)

        """
        if is_distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, shuffle=is_train
            )
        elif is_train:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

        return sampler

    @staticmethod
    def create_optimizer_lrs(config, params):
        """create optimizer and learning rate scheduler.

        Args:
            config: (CfgNode): estimator config:
            params: model parameters
        Returns:
            optimizer: pytorch optimizer
            lr_scheduler: pytorch LR scheduler

        """
        if config.optimizer.name == "Adam":
            optimizer = torch.optim.Adam(params, **config.optimizer.args)

            # use fixed learning rate when using Adam
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda x: 1.0
            )
        elif config.optimizer.name == "SGD":
            optimizer = torch.optim.SGD(
                params,
                lr=config.optimizer.args.lr,
                momentum=config.optimizer.args.momentum,
                weight_decay=float(config.optimizer.args.weight_decay),
            )
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.optimizer.args.lr_step_size,
                gamma=config.optimizer.args.lr_gamma,
            )
        else:
            raise ValueError(
                f"only valid optimizers are 'SGD' and 'Adam' but "
                f"received paramater {config.optimizer.name}"
            )
        return optimizer, lr_scheduler

    @staticmethod
    def get_transform():
        """transform bounding box and tesnor."""
        transforms = [BoxListToTensor(), ToTensor()]
        return Compose(transforms)

    @staticmethod
    def collate_fn(batch):
        """Prepare batch to be format Faster RCNN expects.

        Args:
            batch: mini batch of the form ((x0,x1,x2...xn),(y0,y1,y2...yn))

        Returns:
            mini batch in the form [(x0,y0), (x1,y2), (x2,y2)... (xn,yn)]
        """
        return tuple(zip(*batch))


def create_dataset(config, data_path, split):
    """download dataset from source.

    Args:
        config: (CfgNode): estimator config:
        data_path: Directory on localhost where datasets are located.
        split: train, val, test

    Returns dataset: dataset obj must have len and __get_item__

    """
    dataset = Dataset.create(
        config[split].dataset.name,
        data_path=data_path,
        transforms=FasterRCNN.get_transform(),
        **config[split].dataset.args,
    )
    return dataset


def create_dataloader(
    distributed,
    dataset,
    sampler,
    train,
    *,
    batch_size=1,
    num_workers=0,
    collate_fn=None,
):
    """load dataset and create dataloader.

    Args:
        distributed: wether or not the dataloader is distributed
        dataset: dataset obj must have len and __get_item__
        sampler: (torch.utils.data.Sampler)
        train: whether or not the sampler is for training data
        batch_size: batch_size
        num_workers: num_workers
        collate_fn: Prepare batch to be format Faster RCNN expects
    Returns data_loader:
        torch.utils.data.DataLoader

    """
    if distributed:
        if train:
            batch_sampler = torch.utils.data.BatchSampler(
                sampler, batch_size, drop_last=False
            )

            loader = torch.utils.data.DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=num_workers,
                collate_fn=collate_fn,
            )
            return loader
        else:
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                collate_fn=collate_fn,
                drop_last=False,
            )
            return loader
    else:
        logger.info(f"not creating distributed dataloader")
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            drop_last=train,
            collate_fn=collate_fn,
        )
        return data_loader


def dataloader_creator(config, dataset, sampler, split, distributed):
    """initiate data loading.

    Args:
        config: (CfgNode): estimator config:
        dataset: dataset obj must have len and __get_item__
        sampler: (torch.utils.data.Sampler)
        split: train, val, test
        distributed:
    Returns data_loader:
        torch.utils.data.DataLoader

    """
    is_train = False
    if split == TRAIN:
        is_train = True
    dataloader = create_dataloader(
        distributed=distributed,
        dataset=dataset,
        batch_size=config[split].batch_size,
        sampler=sampler,
        collate_fn=FasterRCNN.collate_fn,
        train=is_train,
    )
    return dataloader


class BadLoss(Exception):
    """pass the exception."""

    pass


class Loss:
    """Record Loss during epoch."""

    def __init__(self):
        """initiate loss."""
        self._sum = 0
        self.num_examples = 0

    def reset(self):
        """reset loss."""
        self._sum = 0
        self.num_examples = 0

    def update(self, avg_loss, batch_size):
        """update loss."""
        self._sum += avg_loss * batch_size
        self.num_examples += batch_size

    def compute(self):
        """compute avg loss.

        Returns (float): avg. loss
        """
        if self.num_examples == 0:
            raise ValueError(
                "Loss must have at least one example "
                "before it can be computed."
            )
        return self._sum / self.num_examples


def convert_bboxes2canonical(bboxes):
    """convert bounding boxes to canonical.

    convert bounding boxes from the format used by pytorch torchvision's
    faster rcnn model into our canonical format, a list of list of BBox2Ds.
    Faster RCNN format:
    https://github.com/pytorch/vision/blob/master/torchvision/models/
    detection/faster_rcnn.py#L45
    Args:
        bboxes (List[Dict[str, torch.Tensor()): A list of dictionaries. Each
        item in the list corresponds to the bounding boxes for one example.
        The dictionary must have the keys 'boxes' and 'labels'. The value for
        'boxes' is (``FloatTensor[N, 4]``): the ground-truth boxes in
        ``[x1, y1, x2, y2]`` format, with values between ``0`` and ``H`` and
        ``0`` and ``W``. The value for labels is (``Int64Tensor[N]``): the
        class label for each ground-truth box. If the dictionary has the key
        `scores` then these values are used for the confidence score of the
        BBox2D, otherwise the score is set to 1.

    Returns (list[List[BBox2D]]):
        Each element in the list corresponds to the
        list of bounding boxes for an example.

    """
    bboxes_batch = []
    for example in bboxes:
        all_coords = example["boxes"]
        num_boxes = all_coords.shape[0]
        labels = example["labels"]
        if "scores" in example.keys():
            scores = example["scores"]
        else:
            scores = torch.FloatTensor([1.0] * num_boxes)
        bboxes_example = []
        for i in range(num_boxes):
            coords = all_coords[i]
            x, y = coords[0].item(), coords[1].item()
            canonical_box = BBox2D(
                x=x,
                y=y,
                w=coords[2].item() - x,
                h=coords[3].item() - y,
                label=labels[i].item(),
                score=scores[i].item(),
            )
            bboxes_example.append(canonical_box)
        bboxes_batch.append(bboxes_example)
    return bboxes_batch


def reduce_dict(input_dict, average=True):
    """Reduce the values in dictionary.

    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results.
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum

    Return:
        dict with the same fields as input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def prepare_bboxes(bboxes: List[BBox2D]) -> Dict[str, torch.Tensor]:
    """Prepare bounding boxes for model training.

    Args:
        bboxes: mini batch of bounding boxes (not including images).
        Each example is a list of bounding boxes. Torchvision's implementation
        of Faster-RCNN requires bounding boxes to be in the format of a
        dictionary {'labels':[label ids, ...], and
        'boxes': [[xleft, ytop, xright, ybottom of box1],
        [xleft, ytop, xright, ybottom of box2]...]

    Returns:
        bounding boxes in the form that Faster RCNN expects
    """
    labels = []
    box_list = []
    for b in bboxes:
        labels.append(b.label)
        box = [b.x, b.y, b.x + b.w, b.y + b.h]
        if box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError(
                f"box not properly formed with coordinates: "
                f"{box} [xleft, ytop, xright, ybottom]"
            )
        box_list.append(box)

    d = {"boxes": torch.Tensor(box_list), "labels": torch.LongTensor(labels)}
    return d


def canonical2list(bbox: BBox2D):
    """convert a BBox2d into a single list.

    Args:
        bbox:

    Returns:
        attribute list of BBox2D

    """
    return [bbox.label, bbox.score, bbox.x, bbox.y, bbox.w, bbox.h]


def list2canonical(box_list):
    """convert a list into a Bbox2d.

    Args:
        box_list: box represented in list format

    Returns:
        BBox2d

    """
    return BBox2D(
        label=box_list[0],
        score=box_list[1],
        x=box_list[2],
        y=box_list[3],
        w=box_list[4],
        h=box_list[5],
    )


def list3d_2canonical(batch):
    """convert 3d list to canonical.

    convert a list of list of padded targets and predictions per examples where
    bounding boxes are represented by lists into the same format except
    the boxes are represented by the BBox2d class and the padded boxes are
    removed.
    Args:
        batch: [[[gt], [prds]], [[gt], [prds]], ] where gt and prds are list
        of lists

    Returns:
        [([gt],[preds]), ([gt],[preds])... where gt and preds are lists of
        BBox2ds
    """
    all_canonical = []
    for example in batch:
        targets, preds = example
        # todo should break after first nan
        targets_canonnical = [
            list2canonical(target_list)
            for target_list in targets
            if not np.isnan(target_list).any()
        ]
        preds = [
            list2canonical(pred_list)
            for pred_list in preds
            if not np.isnan(pred_list).any()
        ]
        all_canonical.append((targets_canonnical, preds))
    return all_canonical


def pad_box_lists(
    gt_preds: List[Tuple[List[BBox2D], List[BBox2D]]],
    max_boxes_per_img=MAX_BOXES_PER_IMAGE,
):
    """Pad the list of boxes.

    Pad the list of boxes and targets with place holder boxes so that all
    targets and predictions have the same number of elements.
    Args:
        gt_preds (list(tuple(list(BBox2d), (Bbox2d)))): A list of tuples where
        the first element in each tuple is a list of bounding boxes
        corresponding to the targets in an example, and the second element
        in the tuple corresponds to the predictions in that example
        max_boxes_per_img: : maximum number of target boxes and predicted boxes
         per image

    Returns: same format as gt_preds but all examples will have the same
    number of targets and predictions. If there are fewer targets or
    predictions than max_boxes_per_img, then boxes with nan values are added.

    """
    padding_box = BBox2D(
        label=np.nan, score=np.nan, x=np.nan, y=np.nan, w=np.nan, h=np.nan
    )
    for tup in gt_preds:
        target_list, pred_list = tup
        if len(target_list) > max_boxes_per_img:
            raise ValueError(
                f"max boxes per image set to {max_boxes_per_img},"
                f" but there were {len(target_list)} targets"
                f" found."
            )
        if len(pred_list) > max_boxes_per_img:
            raise ValueError(
                f"max boxes per image set to {max_boxes_per_img},"
                f" but there were {len(target_list)} predictions"
                f" found."
            )
        for i in range(max_boxes_per_img - len(target_list)):
            target_list.append(padding_box)
        for i in range(max_boxes_per_img - len(pred_list)):
            pred_list.append(padding_box)
    return gt_preds


def _gt_preds2tensor(gt_preds, max_boxes=MAX_BOXES_PER_IMAGE):
    """convert prediction result to tensor.

    Args:
        gt_preds (list(tuple(list(BBox2d), (Bbox2d)))): A list of tuples where
        the first element in each tuple is a list of bounding boxes
        corresponding to the targets in an example, and the second element
        in the tuple corresponds to the predictions in that example
        max_boxes: maximum number of target boxes and predicted boxes per image

    Returns (Tensor): [[gt, preds], [gt, preds]] where gt and preds are both
    tensors where each element is a 1-d tensor representing a bbox

    """
    gt_preds_padded = pad_box_lists(gt_preds, max_boxes)
    padded_list = []
    for i in range(len(gt_preds_padded)):
        gt_list, target_list = gt_preds_padded[i]
        gt_tensor, target_tensor = (
            [canonical2list(b) for b in gt_list],
            [canonical2list(b) for b in target_list],
        )
        gt_preds_tensor = [gt_tensor, target_tensor]
        padded_list.append(gt_preds_tensor)
    gt_preds_tensor = torch.Tensor(padded_list)
    return gt_preds_tensor


def gather_gt_preds(*, gt_preds, device, max_boxes=MAX_BOXES_PER_IMAGE):
    """gather list of prediction.

    Args:
        gt_preds (list(tuple(list(BBox2d), (Bbox2d)))): A list of tuples where
        the first element in each tuple is a list of bounding boxes
        corresponding to the targets in an example, and the second element
        in the tuple corresponds to the predictions in that example
        device:
        max_boxes: the maximum number of boxes allowed for either targets or
        predictions per image

    Returns (list(tuple(list(BBox2d), (Bbox2d)))): a list in the same format
    as gt_preds but containing all the information across processes e.g. if
    rank 0 has gt_preds = [([box_0], []), ([box_1], [box_2, box_2.5])] and
    rank 1 has gt_preds [([], [box_3]), ([box_4, box_5], [])] then this
    function will return [([box_0], []), ([box_1], [box_2, box_2.5]), ([],
    [box_3]), ([box_4, box_5], [])] the returned list is consistent across all
    processes
    """
    gt_preds_tensor = _gt_preds2tensor(gt_preds, max_boxes)

    all_preds_gt = [
        torch.empty(list(gt_preds_tensor.size())).to(device)
        for i in range(get_world_size())
    ]
    dist.all_gather(tensor_list=all_preds_gt, tensor=gt_preds_tensor.to(device))
    all_preds_gt_canonical = tensorlist2canonical(all_preds_gt)
    return all_preds_gt_canonical


def tensorlist2canonical(tensor_list):
    """convert tensorlist to canonical.

    Converts the gt and predictions into the canonical format and removes
    the boxes with nan values that were added for padding.
    Args:
        tensor_list: [tensor([[gt, prds]), tensor([gt, prds])], ...]

    Returns (list(tuple(list(BBox2d), (Bbox2d)))): A list of tuples where
        the first element in each tuple is a list of bounding boxes
        corresponding to the targets in an example, and the second element
        in the tuple corresponds to the predictions in that example

    """
    gt_preds_canonical = []
    for t in tensor_list:
        gt_preds_canonical += list3d_2canonical(t.tolist())
    return gt_preds_canonical


def metric_per_class_plot(metric_name, data, label_mappings, figsize=(20, 10)):
    """Bar plot for metric per class.

    Args:
        metric_name (str): metric name.
        data (dict): a dictionary of metric per label.
        label_mappings (dict): a dict of {label_id: label_name} mapping
        figsize (tuple): figure size of the plot. Default is (20, 10)

    Returns (matplotlib.pyplot.figure):
        a bar plot for metric per class.
    """
    label_id = [k for k in sorted(data.keys()) if k in label_mappings]
    metric_values = [data[i] for i in label_id]
    label_name = [label_mappings[i] for i in label_id]
    fig = plt.figure(figsize=figsize)
    plt.bar(label_id, metric_values)
    plt.title(f"{metric_name} per class")
    plt.xlabel("label name")
    plt.ylabel(f"{metric_name}")
    plt.xticks(
        label_id, label_name, rotation="vertical",
    )
    plt.margins(0.2)
    plt.subplots_adjust(bottom=0.3)
    return fig


class BoxListToTensor:
    """transform to bboxes to Tensor."""

    def __call__(self, image, target):
        """transform target to bboxes."""
        target = prepare_bboxes(target)
        return image, target


class ToTensor:
    """transform to tesnor."""

    def __call__(self, image, target):
        """transform image to tesnor."""
        image = torchvision.transforms.functional.to_tensor(image)
        return image, target
