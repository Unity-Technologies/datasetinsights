import copy
import logging

import numpy as np
import torch
import torchvision
from ignite.metrics import Loss
from torchvision import transforms as T
from torchvision.transforms import functional as F

from datasetinsights.datasets import Dataset
from datasetinsights.evaluation_metrics import EvaluationMetric
from datasetinsights.io.loader import create_loader
from datasetinsights.io.transforms import Compose, RandomHorizontalFlip
from datasetinsights.stats.visualization.plots import decode_segmap, grid_plot

from .base import Estimator

logger = logging.getLogger(__name__)

# Normalization constants (heuristics) from ImageNet dataset
_IMGNET_MEAN = (0.485, 0.456, 0.406)
_IMGNET_STD = (0.229, 0.224, 0.225)

# Inverse Normalization constants
_INV_IMGNET_MEAN = (-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225)
_INV_IMGNET_STD = (1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.5)


def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)

    return img


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)

        return image, target


class ToTensor:
    """Convert a pair of (image, target) to tensor
    """

    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.asarray(target), dtype=torch.int64)

        return image, target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)

        return image, target


class DeeplabV3(Estimator):
    """ DeeplabV3 Model https://arxiv.org/abs/1706.05587

    Args:
        config (CfgNode): estimator config
        writer: Tensorboard writer object
        checkpointer: Model checkpointer callback to save models
        device: model training on device (cpu|cuda)
    Attributes:
        backbone: model backbone (resnet50|resnet101)
        num_classes: number of classes for semantic segmentation
        model: tensorflow or pytorch graph
        writer: Tensorboard writer object
        checkpointer: Model checkpointer callback to save models
        device: model training on device (cpu|cuda)
        optimizer: pytorch optimizer
        lr_scheduler: pytorch learning rate scheduler
    """

    def __init__(
        self,
        *,
        config,
        writer,
        checkpointer,
        device,
        checkpoint_file=None,
        **kwargs,
    ):
        self.config = config
        self.writer = writer
        self.checkpointer = checkpointer
        self.device = device

        self.backbone = config.backbone
        self.num_classes = config.num_classes

        model_name = "deeplabv3_" + config.backbone
        self.model = torchvision.models.segmentation.__dict__[model_name](
            num_classes=self.num_classes
        )

        opname = self.config.optimizer.name
        if opname == "Adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(), **self.config.optimizer.args
            )

            # use fixed learning rate when using Adam
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda x: 1.0
            )
        else:
            raise ValueError(f"Unsupported optimizer type {opname}")

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        # load estimators from file if checkpoint_file exists
        if checkpoint_file:
            self.checkpointer.load(self, checkpoint_file)

    @staticmethod
    def _transforms(is_train=True, crop_size=769):
        """Transformations for a pair of input and target image

        Args:
            is_train (bool): indicator whether this is a transformation
                during training (default: True)
            crop_size (int): crop size. Images will be cropped to
                (crop_size, crop_size)
        """
        transforms = []
        if is_train:
            transforms.append(RandomHorizontalFlip(0.5))
            transforms.append(RandomCrop(crop_size))
        transforms.append(ToTensor())
        transforms.append(Normalize(mean=_IMGNET_MEAN, std=_IMGNET_STD))

        return Compose(transforms)

    @staticmethod
    def _loss_fn(outputs, target):
        """ Compute loss

        Args:
            outputs (dict): named output of deeplabv3 model. Since this
            implementation outpus two semantic segmentation images from two
            heads of the model, we are expecting dict of tow keys
            "out" and "aux" that corresponds to two pytorch tenor of images.
            target (torch.Tensor): ground truth 2D image tensor

        Returns:
            numerical value of loss
        """
        losses = {}
        for name, x in outputs.items():
            losses[name] = torch.nn.functional.cross_entropy(
                x, target, ignore_index=255
            )

        if len(losses) == 1:
            return losses["out"]

        return losses["out"] + 0.5 * losses["aux"]

    def _train_one_epoch(self, loader, epoch):
        """ Train one epoch

        Args:
            loader (DataLoader): pytorch dataloader
            epoch (int): the current epoch number
        """
        logger.info(f"Epoch[{epoch}] training started.")
        self.model.train()
        n_batch = len(loader)
        accumulation_steps = self.config.train.accumulation_steps
        loss_metric = Loss(self._loss_fn)

        self.optimizer.zero_grad()
        for i, (image, target) in enumerate(loader):
            image, target = image.to(self.device), target.to(self.device)
            output = self.model(image)
            loss = self._loss_fn(output, target)
            loss.backward()

            # Accumulated Gradients are only updated after X steps.
            # This creates an effective batch size of
            # batch_size * accumulation_steps
            if (i + 1) % accumulation_steps == 0:
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

            loss_metric.update((output, target))

            iter_num = (i + 1) % n_batch
            logger.debug(
                f"Epoch[{epoch}] Iteration[{iter_num}/{n_batch}] "
                f"Loss: {loss:.3f}"
            )

        epoch_loss = loss_metric.compute()
        logger.info(
            f"Epoch[{epoch}] training completed. Loss: {epoch_loss:.3f}"
        )
        self.writer.add_scalar("training/loss", epoch_loss, epoch)

        loss_metric.reset()

    def _evaluate_one_epoch(self, loader, epoch):
        """ Evaluate one epoch

        Args:
            loader (DataLoader): pytorch dataloader
            epoch (int): the current epoch number
        """
        logger.info(f"Epoch[{epoch}] evaluation started")
        self.model.eval()
        loss_metric = Loss(self._loss_fn)

        # TODO: Support other metrics other than IoU and support multiple
        # mettics
        iou_metric = EvaluationMetric.create(
            self.config.metric, num_classes=self.num_classes
        )
        with torch.no_grad():
            for image, target in loader:
                image, target = image.to(self.device), target.to(self.device)
                output = self.model(image)

                loss_metric.update((output, target))
                iou_metric.update((output["out"], target))

        loss = loss_metric.compute()
        iou = iou_metric.compute()

        # some classes are not used in cityscapes evaluation.
        # TODO: Move class masking logic to IoU metric class.
        keep_mask = [
            not c.ignore_in_eval
            for c in torchvision.datasets.Cityscapes.classes
        ]
        class_names = [c.name for c in torchvision.datasets.Cityscapes.classes]
        iou_info = {
            name: f"{iou[i].item():.3f}"
            for i, name in enumerate(class_names)
            if keep_mask[i]
        }
        miou = iou[keep_mask].mean()

        logger.info(
            f"Epoch[{epoch}] evaluation completed. "
            f"Loss: {loss:.3f}, mIoU: {miou:.3f}\n"
            f"IoU per class: {iou_info}"
        )
        self.writer.add_scalar("validation/loss", loss, epoch)
        self.writer.add_scalar("validation/miou", miou, epoch)

        inv_normalize = T.Normalize(mean=_INV_IMGNET_MEAN, std=_INV_IMGNET_STD)
        # Visualize segmentation images from last mini-batch
        n_images = list(image.shape)[0]
        image_grid = []
        for i in range(n_images):
            img = inv_normalize(image[i, :]).permute(1, 2, 0).cpu().numpy()
            out = decode_segmap(output["out"][i, :].max(0)[1].cpu().numpy())
            tgt = decode_segmap(target[i, :].cpu().numpy())
            image_grid.append([img, out, tgt])

        fig = grid_plot(image_grid)
        self.writer.add_figure("validation/visualize", fig, epoch)

        loss_metric.reset()
        iou_metric.reset()

    def train(self, **kwargs):
        config = self.config
        train_dataset = Dataset.create(
            config.train.dataset,
            split="train",
            data_root=config.system.data_root,
            transforms=self._transforms(
                is_train=True, crop_size=config.train.crop_size
            ),
        )
        train_loader = create_loader(
            train_dataset,
            batch_size=config.train.batch_size,
            num_workers=config.system.workers,
            dryrun=config.system.dryrun,
        )

        val_dataset = Dataset.create(
            config.val.dataset,
            split="val",
            data_root=config.system.data_root,
            transforms=self._transforms(is_train=False),
        )
        val_loader = create_loader(
            val_dataset,
            batch_size=config.val.batch_size,
            num_workers=config.system.workers,
            dryrun=config.system.dryrun,
        )

        logger.info("Start training estimator: %s", type(self).__name__)
        self.model.to(self.device)
        n_epochs = config.train.epochs
        val_interval = config.system.val_interval
        for epoch in range(1, n_epochs + 1):
            logger.info(f"Training Epoch[{epoch}/{n_epochs}]")
            self._train_one_epoch(train_loader, epoch)

            if epoch % val_interval == 0:
                self._evaluate_one_epoch(val_loader, epoch)

            self.checkpointer.save(self, epoch=epoch)

        self.writer.close()

    def evaluate(self, **kwargs):
        config = self.config
        test_dataset = Dataset.create(
            config.test.dataset,
            split="test",
            data_root=config.system.data_root,
            transforms=self._transforms(is_train=False),
        )
        test_loader = create_loader(
            test_dataset,
            batch_size=config.test.batch_size,
            num_workers=config.system.workers,
            dryrun=config.system.dryrun,
        )

        logger.info("Start evaluating estimator: %s", type(self).__name__)
        self.model.to(self.device)
        self._evaluate_one_epoch(test_loader, epoch=1)
        self.writer.close()

    def save(self, path):
        """ Serialize Estimator to path

        Args:
            path (str): full path to save serialized estimator

        Returns:
            saved full path of the serialized estimator
        """
        save_dict = {"model": self.model.state_dict(), "config": self.config}
        torch.save(save_dict, path)

        return path

    def load(self, path):
        """ Load Estimator from path

        Args:
            path (str): full path to the serialized estimator
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model"])

        loaded_config = copy.deepcopy(checkpoint["config"])
        stored_config = copy.deepcopy(self.config)
        del stored_config["checkpoint_file"]
        del loaded_config["checkpoint_file"]
        if stored_config != loaded_config:
            logger.warning(
                f"Found difference in estimator config."
                f"Estimator loaded from {path} was trained using "
                f"config: "
                f"{loaded_config}. However, the current config is: "
                f"{self.config}."
            )
