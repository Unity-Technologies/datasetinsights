import copy
import logging
import math
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torchvision
from codetiming import Timer

import datasetinsights.constants as const
from datasetinsights.data.bbox import BBox2D
from datasetinsights.data.datasets import Dataset
from datasetinsights.data.transforms import Compose
from datasetinsights.evaluation_metrics.base import EvaluationMetric
from datasetinsights.torch_distributed import get_world_size

from .base import Estimator

MAX_BOXES_PER_IMAGE = 100
logger = logging.getLogger(__name__)
DEFAULT_ACCUMULATION_STEPS = 1


class BadLoss(Exception):
    pass


class Loss:
    """
    Record Loss during epoch
    """

    def __init__(self):
        self._sum = 0
        self.num_examples = 0

    def reset(self):
        self._sum = 0
        self.num_examples = 0

    def update(self, avg_loss, batch_size):
        self._sum += avg_loss * batch_size
        self.num_examples += batch_size

    def compute(self):
        """

        Returns (float): avg. loss

        """
        if self.num_examples == 0:
            raise ValueError(
                "Loss must have at least one example "
                "before it can be computed."
            )
        return self._sum / self.num_examples


def convert_bboxes2canonical(bboxes):
    """
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

    Returns (list[List[BBox2D]]): Each element in the list corresponds to the
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
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
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

    Returns: bounding boxes in the form that Faster RCNN expects
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


def train_one_epoch(
    *,
    writer,
    model,
    optimizer,
    data_loader,
    device,
    epoch,
    lr_scheduler,
    accumulation_steps,
    log_frequency=1000,
):
    """

    Args:
        writer: Tensorboard writer object
        model: pytorch model
        optimizer: pytorch optimizer
        data_loader(DataLoader): pytorch dataloader
        device: model training on device (cpu|cuda)
        epoch (int):
        accumulation_steps (int): Accumulated Gradients are only updated after
        X steps. This creates an effective batch size of batch_size *
        accumulation_steps
        lr_scheduler: Pytorch LR scheduler
        log_frequency: after this number of mini batches log average loss
    """
    model.train()
    loss_metric = Loss()
    optimizer.zero_grad()
    n_examples = len(data_loader.dataset)
    for i, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)

        losses_grad = sum(loss for loss in loss_dict.values())

        loss_dict_reduced = reduce_dict(loss_dict)
        losses = sum(loss for loss in loss_dict_reduced.values())
        loss_metric.update(avg_loss=losses.item(), batch_size=len(targets))

        if not math.isfinite(losses):
            raise BadLoss(
                f"Loss is {losses}, stopping training. Input was "
                f"{images, targets}. Loss dict is {loss_dict}"
            )
        elif i % log_frequency == 0:
            intermediate_loss = loss_metric.compute()
            examples_seen = epoch * n_examples + loss_metric.num_examples
            logger.info(
                f"intermediate loss after mini batch {i} in epoch {epoch} "
                f"(total training examples: {examples_seen}) is "
                f"{intermediate_loss}"
            )
            writer.add_scalar(
                "training/running_loss", intermediate_loss, examples_seen
            )
        losses_grad.backward()
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
    writer.add_scalar("training/loss", loss_metric.compute(), epoch)
    writer.add_scalar(
        "training/running_loss", loss_metric.compute(), loss_metric.num_examples
    )
    writer.add_scalar("training/lr", optimizer.param_groups[0]["lr"], epoch)
    loss_metric.reset()


def canonical2list(bbox: BBox2D):
    """
    convert a BBox2d into a single list
    Args:
        bbox:

    Returns:

    """
    return [bbox.label, bbox.score, bbox.x, bbox.y, bbox.w, bbox.h]


def list2canonical(box_list):
    """
    convert a list into a Bbox2d
    Args:
        box_list: box represented in list format

    Returns (BBox2d):

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
    """
    convert a list of list of padded targets and predictions per examples where
    bounding boxes are represented by lists into the same format except
    the boxes are represented by the BBox2d class and the padded boxes are
    removed
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
    """
    Pad the list of boxes and targets with place holder boxes so that all
    targets and predictions have the same number of elements
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
    """

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
    """

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
    """
    Converts the gt and predictions into the canonical format and removes
    the boxes with nan values that were added for padding
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
    """
    Bar plot for metric per class.
    Args:
        metric_name (str): metric name.
        data (dict): a dictionary of metric per label.
        label_mappings (dict): a dict of {label_id: label_name} mapping
        figsize (tuple): figure size of the plot. Default is (20, 10)

    Returns (matplotlib.pyplot.figure): a bar plot for metric per class.
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


@torch.no_grad()
def evaluate_per_epoch(
    *,
    model,
    data_loader,
    device,
    writer,
    kfp_writer,
    epoch,
    metrics,
    label_mappings,
    max_detections_per_img=MAX_BOXES_PER_IMAGE,
    is_distributed=False,
    synchronize_metrics=True,
):
    """
    Evaluate model performance. Note, torchvision's implementation of faster
    rcnn requires input and gt data for training mode and returns a dictionary
    of losses (which we need to record the loss). We also need to get the raw
    predictions, which is only possible in model.eval() mode, to calculate
    the evaluation metric.
    Args:
        model: pytorch model
        data_loader (DataLoader): pytorch dataloader
        device: model training on device (cpu|cuda)
        writer: Tensorboard writer object
        epoch (int): current epoch, used for logging
        metrics (dict): a dict of metrics (key: metric name; value: metric)
        label_mappings (dict): a dict of {label_id: label_name} mapping
        max_detections_per_img: max number of targets or predictions allowed
        per example
        is_distributed: whether or not the model is distributed
        synchronize_metrics: whether or not to synchronize evaluation metrics
        across processes
    Returns:

    """
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    model.eval()

    loss_metric = Loss()
    for metric in metrics.values():
        metric.reset()
    for i, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets_raw = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets_converted = convert_bboxes2canonical(targets)
        # we need 2 do forward passes, one in eval mode gets us the raw preds
        # and one in train mode gets us the losses
        # pass number 1 to get mAP
        model.eval()
        preds_raw = model(images)
        converted_preds = convert_bboxes2canonical(preds_raw)
        gt_preds = list(zip(targets_converted, converted_preds))
        if is_distributed and synchronize_metrics:
            all_preds_gt_canonical = gather_gt_preds(
                gt_preds=gt_preds,
                device=device,
                max_boxes=max_detections_per_img,
            )
        else:
            all_preds_gt_canonical = gt_preds
        for metric in metrics.values():
            metric.update(all_preds_gt_canonical)

        # pass number 2 to get val loss
        model.train()
        loss_dict = model(images, targets_raw)
        loss_dict_reduced = reduce_dict(loss_dict)
        losses = sum(loss for loss in loss_dict_reduced.values())
        loss_metric.update(avg_loss=losses.item(), batch_size=len(targets))

    for metric_name, metric in metrics.items():
        result = metric.compute()
        logger.debug(result)
        mean_result = np.mean(
            [result_per_label for result_per_label in result.values()]
        )
        logger.info(f"metric {metric_name} has mean result: {mean_result}")
        writer.add_scalar(f"val/{metric_name}", mean_result, epoch)

        kfp_writer.add_metric(name=metric_name, val=mean_result)
        # TODO (YC) This is hotfix to allow user map between label_id
        # to label_name during model evaluation. In ideal cases this mapping
        # should be available before training/evaluation dataset was loaded.
        # label_id that was missing from label_name should be removed from
        # dataset and the training procedure.
        label_results = {
            label_mappings.get(id, str(id)): value
            for id, value in result.items()
        }
        writer.add_scalars(
            f"val/{metric_name}-per-class", label_results, epoch
        )
        fig = metric_per_class_plot(metric_name, result, label_mappings)
        writer.add_figure(f"{metric_name}-per-class", fig, epoch)

    val_loss = loss_metric.compute()
    logger.info(f"validation loss is {val_loss}")
    writer.add_scalar("val/loss", val_loss, epoch)

    torch.set_num_threads(n_threads)


class PrepBoxes:
    def __call__(self, image, target):
        target = prepare_bboxes(target)
        return image, target


class ToTensor:
    def __call__(self, image, target):
        image = torchvision.transforms.functional.to_tensor(image)
        return image, target


def get_transform():
    transforms = [PrepBoxes(), ToTensor()]
    return Compose(transforms)


def collate_fn(batch):
    """
    Prepare batch to be format Faster RCNN expects
    Args:
        batch: mini batch of the form ((x0,x1,x2...xn),(y0,y1,y2...yn))

    Returns:
        mini batch in the form [(x0,y0), (x1,y2), (x2,y2)... (xn,yn)]
    """
    return tuple(zip(*batch))


class FasterRCNN(Estimator):
    """
    https://github.com/pytorch/vision/tree/master/references/detection
    https://arxiv.org/abs/1506.01497
    Args:
        config (CfgNode): estimator config
        writer: Tensorboard writer object
        checkpointer: Model checkpointer callback to save models
        device: model training on device (cpu|cuda)
        local_rank (int): (optional) rank of process executing code
        gpu (int): (optional) gpu id on which code will execute
    Attributes:
        model: pytorch model
        writer: Tensorboard writer object
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
        device,
        box_score_thresh=0.05,
        gpu=0,
        rank=0,
        **kwargs,
    ):
        logger.info(f"initializing faster rcnn")
        self.config = config
        self.device = device
        self.writer = writer
        self.kfp_writer = kfp_writer
        model_name = f"fasterrcnn_{self.config.backbone}_fpn"
        self.model = torchvision.models.detection.__dict__[model_name](
            num_classes=config.num_classes,
            pretrained_backbone=config.pretrained_backbone,
            pretrained=config.pretrained,
            box_detections_per_img=MAX_BOXES_PER_IMAGE,
            box_score_thresh=box_score_thresh,
        )
        self.model_without_ddp = self.model
        self.gpu = gpu
        self.rank = rank
        self.sync_metrics = config.get("synchronize_metrics", True)
        logger.info(f"gpu: {self.gpu}, rank: {self.rank}")

        self.checkpointer = checkpointer
        checkpoint_file = config.get("checkpoint_file", const.NULL_STRING)
        if checkpoint_file != const.NULL_STRING:
            checkpointer.load(self, config.checkpoint_file)

        self.metrics = {}
        for metric_key, metric in config.metrics.items():
            self.metrics[metric_key] = EvaluationMetric.create(
                metric.name, **metric.args
            )
        self.model.to(self.device)

        if self.config.system.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.gpu]
            )
            self.model_without_ddp = self.model.module

    def evaluate(self, **kwargs):
        config = self.config
        test_dataset = Dataset.create(
            config.test.dataset.name,
            data_root=config.system.data_root,
            transforms=get_transform(),
            **config.test.dataset.args,
        )

        label_mappings = test_dataset.label_mappings
        if self.config.system.dryrun:
            test_dataset = self._create_dryrun_dataset(
                dataset=test_dataset, subset_size=self.config.val.batch_size * 2
            )
        test_sampler = self._create_sampler(
            dataset=test_dataset, is_train=False,
        )

        logger.info("Start evaluating estimator: %s", type(self).__name__)
        test_loader = self._create_loader(
            dataset=test_dataset,
            batch_size=self.config.val.batch_size,
            sampler=test_sampler,
            collate_fn=collate_fn,
            train=False,
        )
        self.model.to(self.device)
        evaluate_per_epoch(
            model=self.model,
            data_loader=test_loader,
            device=self.device,
            writer=self.writer,
            kfp_writer=self.kfp_writer,
            epoch=0,
            metrics=self.metrics,
            label_mappings=label_mappings,
            is_distributed=self.config.system.distributed,
            synchronize_metrics=self.sync_metrics,
        )

    def _create_sampler(self, *, dataset, is_train):
        """

        Args:
            dataset: dataset obj must have len and __get_item__
            is_train: whether or not the sampler is for training data

        Returns:
            https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler

        """
        if self.config.system.distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, shuffle=is_train
            )

        else:
            if is_train:
                sampler = torch.utils.data.RandomSampler(dataset)
            else:
                sampler = torch.utils.data.SequentialSampler(dataset)
        return sampler

    def _create_loader(
        self,
        dataset,
        sampler,
        train,
        *,
        batch_size=1,
        num_workers=0,
        collate_fn=None,
    ):
        if self.config.system.distributed:
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

    def _create_dryrun_dataset(self, dataset, subset_size):
        r = np.random.default_rng()
        indices = r.integers(0, len(dataset), subset_size)
        dataset = torch.utils.data.Subset(dataset, indices)
        return dataset

    def train(self, **kwargs):
        config = self.config
        train_dataset = Dataset.create(
            config.train.dataset.name,
            data_root=config.system.data_root,
            transforms=get_transform(),
            **config.train.dataset.args,
        )
        logger.info(f"length of train dataset is {len(train_dataset)}")
        val_dataset = Dataset.create(
            config.val.dataset.name,
            data_root=config.system.data_root,
            transforms=get_transform(),
            **config.val.dataset.args,
        )
        logger.info(f"length of validation dataset is {len(val_dataset)}")

        label_mappings = train_dataset.label_mappings
        if self.config.system.dryrun:
            train_dataset = self._create_dryrun_dataset(
                dataset=train_dataset,
                subset_size=self.config.train.batch_size * 2,
            )
            val_dataset = self._create_dryrun_dataset(
                dataset=val_dataset, subset_size=self.config.val.batch_size * 2
            )
        train_sampler = self._create_sampler(
            dataset=train_dataset, is_train=True,
        )
        val_sampler = self._create_sampler(dataset=val_dataset, is_train=False)

        train_loader = self._create_loader(
            dataset=train_dataset,
            batch_size=self.config.train.batch_size,
            sampler=train_sampler,
            collate_fn=collate_fn,
            train=True,
        )
        val_loader = self._create_loader(
            dataset=val_dataset,
            batch_size=self.config.val.batch_size,
            sampler=val_sampler,
            collate_fn=collate_fn,
            train=False,
        )
        self.train_loop(
            config=self.config,
            train_dataloader=train_loader,
            label_mappings=label_mappings,
            val_dataloader=val_loader,
            writer=self.writer,
            kfp_writer=self.kfp_writer,
            model=self.model,
            train_sampler=train_sampler,
        )

    def save(self, path):
        """ Serialize Estimator to path

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
        """ Load Estimator from path

        Args:
            path (str): full path to the serialized estimator
        """
        logger.info(f"loading checkpoint from file")
        checkpoint = torch.load(path, map_location=self.device)
        self.model_without_ddp.load_state_dict(checkpoint["model"])
        loaded_config = copy.deepcopy(checkpoint["config"])
        stored_config = copy.deepcopy(self.config)
        del stored_config["checkpoint_file"]
        if stored_config != loaded_config:
            logger.debug(
                f"Found difference in estimator config."
                f"Estimator loaded from {path} was trained using config: \n"
                f"{loaded_config}. \nHowever, the current config is: \n"
                f"{self.config}."
            )
        self.model_without_ddp.eval()

    def predict(self, pil_img, box_score_thresh=0.5):
        """ Get prediction from one image using loaded model

        Args:
            pil_img (PIL Image): PIL image from dataset.
            box_score_thresh (float): box score threshold for filter out lower
            score bounding boxes. Defaults to 0.5.

        Returns:
            filtered_pred_annotations (List[BBox2D]): high predicted score
            bboxes from the model.
        """
        pil_img = pil_img.unsqueeze(0)
        img_tensor = pil_img.to(self.device)
        predicts = self.model_without_ddp(img_tensor)
        predict_annotations = convert_bboxes2canonical(predicts)[0]
        filtered_pred_annotations = [
            box for box in predict_annotations if box.score >= box_score_thresh
        ]
        return filtered_pred_annotations

    def train_loop(
        self,
        *,
        model,
        config,
        writer,
        kfp_writer,
        train_dataloader,
        label_mappings,
        val_dataloader,
        train_sampler=None,
    ):
        """

        Args:
            model: model to train
            config (CfgNode): estimator config
            writer: Tensorboard writer object
            device: model training on device (cpu|cuda)
            train_dataloader (torch.utils.data.DataLoader):
            label_mappings (dict): a dict of {label_id: label_name} mapping
            val_dataloader (torch.utils.data.DataLoader):
        """
        model = model.to(self.device)
        params = [p for p in model.parameters() if p.requires_grad]
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

        accumulation_steps = config.train.get(
            "accumulation_steps", DEFAULT_ACCUMULATION_STEPS
        )
        logger.debug("Start training")
        total_timer = Timer(
            name="total-time", text=const.TIMING_TEXT, logger=logging.info
        )
        total_timer.start()
        for epoch in range(config.train.epochs):
            with Timer(
                name=f"epoch-{epoch}-train-time",
                text=const.TIMING_TEXT,
                logger=logging.info,
            ):
                train_one_epoch(
                    writer=writer,
                    model=model,
                    optimizer=optimizer,
                    data_loader=train_dataloader,
                    device=self.device,
                    epoch=epoch,
                    lr_scheduler=lr_scheduler,
                    accumulation_steps=accumulation_steps,
                    log_frequency=self.config.train.log_frequency,
                )
            if self.config.system.distributed:
                train_sampler.set_epoch(epoch)
            self.checkpointer.save(self, epoch=epoch)
            with Timer(
                name=f"epoch-{epoch}-evaluate-time",
                text=const.TIMING_TEXT,
                logger=logging.info,
            ):
                evaluate_per_epoch(
                    model=model,
                    data_loader=val_dataloader,
                    device=self.device,
                    writer=writer,
                    kfp_writer=kfp_writer,
                    epoch=epoch,
                    metrics=self.metrics,
                    label_mappings=label_mappings,
                )
        total_timer.stop()
