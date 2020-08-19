import copy
import logging
import random
from itertools import permutations

import kornia
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from ignite.metrics import Loss
from PIL import Image
from torchvision.transforms import Compose
from torchvision.transforms.functional import to_tensor

from datasetinsights.datasets import Dataset
from datasetinsights.evaluation_metrics import EvaluationMetric
from datasetinsights.io.loader import create_loader
from datasetinsights.io.transforms import RandomHorizontalFlip, Resize
from datasetinsights.stats.visualization.plots import grid_plot

from .base import Estimator

logger = logging.getLogger(__name__)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)


class RandomChannelSwap:
    """Swap color channel of the image

    Args:
        probability: the probability to swap color channel of the image
    """

    def __init__(self, probability):
        self.probability = probability

    def __call__(self, sample):
        image, depth = sample

        if random.random() < self.probability:
            image = np.asarray(image)
            perm = random.choice(list(permutations(range(3))))
            image = Image.fromarray(image[..., perm])

        return (image, depth)


class ToTensor(object):
    """Convert the image and depth to tensor."""

    def __call__(self, sample):
        image, depth = sample
        image = to_tensor(image)
        depth = to_tensor(depth)

        return (image, depth)


class UpSample(nn.Sequential):
    def __init__(self, skip_input, output_features):
        super().__init__()
        self.convA = nn.Conv2d(
            skip_input, output_features, kernel_size=3, stride=1, padding=1
        )
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(
            output_features, output_features, kernel_size=3, stride=1, padding=1
        )
        self.leakyreluB = nn.LeakyReLU(0.2)

    def forward(self, x, concat_with):
        up_x = F.interpolate(
            x,
            size=[concat_with.size(2), concat_with.size(3)],
            mode="bilinear",
            align_corners=True,
        )
        return self.leakyreluB(
            self.convB(
                self.leakyreluA(
                    self.convA(torch.cat([up_x, concat_with], dim=1))
                )
            )
        )


class Decoder(nn.Module):
    def __init__(self, num_features=2208, decoder_width=0.5):
        super().__init__()
        features = int(num_features * decoder_width)

        self.conv2 = nn.Conv2d(
            num_features, features, kernel_size=1, stride=1, padding=1
        )

        self.up1 = UpSample(
            skip_input=features // 1 + 384, output_features=features // 2
        )
        self.up2 = UpSample(
            skip_input=features // 2 + 192, output_features=features // 4
        )
        self.up3 = UpSample(
            skip_input=features // 4 + 96, output_features=features // 8
        )
        self.up4 = UpSample(
            skip_input=features // 8 + 96, output_features=features // 16
        )

        self.conv3 = nn.Conv2d(
            features // 16, 1, kernel_size=3, stride=1, padding=1
        )

    def forward(self, features):
        x_block0 = features[3]
        x_block1 = features[4]
        x_block2 = features[6]
        x_block3 = features[8]
        x_block4 = features[11]
        x_d0 = self.conv2(x_block4)
        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)
        return self.conv3(x_d4)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.original_model = models.densenet161(pretrained=True)

    def forward(self, x):
        features = [x]
        for k, v in self.original_model.features._modules.items():
            features.append(v(features[-1]))
        return features


class DenseDepthModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        return self.decoder(self.encoder(x))


class DenseDepth(Estimator):
    """
    Attributes:
        config: estimator config
        writer: Tensorboard writer object
        model: tensorflow or pytorch graph
        checkpointer: Model checkpointer callback to save models
        device: model training on device (cpu|cuda)
        optimizer: pytorch optimizer
    """

    def __init__(
        self, config, writer, checkpointer, device, checkpoint_file, **kwargs
    ):
        """
        Args:
        config: estimator config
        writer: Tensorboard writer object
        checkpointer: Model checkpointer callback to save models
        device: model training on device (cpu|cuda)
        """
        # initialize new model or load pre-trained model
        self.config = config
        self.checkpointer = checkpointer
        self.device = device
        self.writer = writer

        self.model = DenseDepthModel()
        logger.info("DenseDepth model is created.")

        opname = self.config.optimizer.name

        if opname == "Adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(), self.config.optimizer.lr
            )
        else:
            raise ValueError(f"Unsupported optimizer type {opname}")

        self.optimizer = optimizer

        # load estimators from file if checkpoint_file exists
        if checkpoint_file:
            self.checkpointer.load(self, checkpoint_file)

    @staticmethod
    def _NYU_transforms(is_train=True):
        transforms = []
        if is_train:
            transforms.append(RandomHorizontalFlip(0.5))
            transforms.append(RandomChannelSwap(0.25))
        transforms.append(Resize(target_size=(240)))
        transforms.append(ToTensor())
        return Compose(transforms)

    @staticmethod
    def _loss_fn(output, depth_n):
        """ Compute loss

        Args:
            output (torch.Tensor): predicted 2D tensor
            depth_n (torch.Tensor): ground truth 2D depth tensor

        Returns:
            numerical value of loss
        """
        _, _, width, height = output.size()
        l1_criterion = nn.L1Loss()
        l_depth = l1_criterion(output, depth_n)
        #  window size = 11 from original repo
        ssim_criterion = kornia.losses.SSIM(min(11, width, height))
        l_ssim = ssim_criterion(output, depth_n).mean().item()
        loss = (1.0 * l_ssim) + (0.1 * l_depth)

        return loss

    def _evaluate_one_epoch(self, loader, epoch, n_epochs):
        """ Evaluate one epoch

        Args:
            loader (DataLoader): pytorch dataloader
            epoch (int): the current epoch number
            n_epochs (int): total epoch number
        """
        logger.info(f"Epoch[{epoch}/{n_epochs}] evaluation started")
        self.model.eval()
        loss_metric = Loss(self._loss_fn)

        # metrics
        # The RMSE is smaller than the normal scale. Because we didn't do the:
        # depth = self.to_tensor(depth).float() * 10
        are_metric = EvaluationMetric.create("AverageRelativeError")
        log10_metric = EvaluationMetric.create("AverageLog10Error")
        rmse_metric = EvaluationMetric.create("RootMeanSquareError")
        a1_metric = EvaluationMetric.create("ThresholdAccuracy", threshold=1.25)
        a2_metric = EvaluationMetric.create(
            "ThresholdAccuracy", threshold=1.25 ** 2
        )
        a3_metric = EvaluationMetric.create(
            "ThresholdAccuracy", threshold=1.25 ** 3
        )

        with torch.no_grad():
            for image, depth in loader:

                image = image.to(self.device)
                depth_n = depth.to(self.device)

                output = self.model(image)

                loss_metric.update((output, depth_n))
                depth_image_pair = (output.cpu().numpy(), depth_n.cpu().numpy())
                are_metric.update(depth_image_pair)
                log10_metric.update(depth_image_pair)
                rmse_metric.update(depth_image_pair)
                a1_metric.update(depth_image_pair)
                a2_metric.update(depth_image_pair)
                a3_metric.update(depth_image_pair)

        # Compute the loss
        loss = loss_metric.compute()
        are_metric_val = are_metric.compute()
        log10_metric_val = log10_metric.compute()
        rmse_metric_val = rmse_metric.compute()
        a1_metric_val = a1_metric.compute()
        a2_metric_val = a2_metric.compute()
        a3_metric_val = a3_metric.compute()

        logger.info(
            f"Epoch[{epoch}/{n_epochs}] evaluation completed.\n"
            f"Validation Loss: {loss:.3f}\n"
            f"Average Relative Error: {are_metric_val:.3f}\n"
            f"Average Log10 Error: {log10_metric_val:.3f}\n"
            f"Root Mean Square Error: {rmse_metric_val:.3f}\n"
            f"Threshold Accuracy (delta1): {a1_metric_val:.3f}\n"
            f"Threshold Accuracy (delta2): {a2_metric_val:.3f}\n"
            f"Threshold Accuracy (delta3): {a3_metric_val:.3f}\n"
        )

        self.writer.add_scalar("Validation/loss", loss, epoch)
        self.writer.add_scalar(
            "Validation/Average_Relative_Error", are_metric_val, epoch
        )
        self.writer.add_scalar(
            "Validation/Average_Log10_Error", log10_metric_val, epoch
        )
        self.writer.add_scalar(
            "Validation/Root_Mean_Square_Error", rmse_metric_val, epoch
        )
        self.writer.add_scalar(
            "Validation/Threshold_Accuracy__delta1_", a1_metric_val, epoch
        )
        self.writer.add_scalar(
            "Validation/Threshold_Accuracy__delta2_", a2_metric_val, epoch
        )
        self.writer.add_scalar(
            "Validation/Threshold_Accuracy__delta3_", a3_metric_val, epoch
        )

        # Visualize depth images from last mini-batch
        n_images = image.shape[0]
        image_grid = []
        gray_depth_grid = []
        for i in range(n_images):
            rgb_image = image[i].permute(1, 2, 0)
            image_grid.append([rgb_image.cpu().numpy()])
            gray_depth_grid.append(
                [
                    output[i].permute(1, 2, 0)[:, :, -1].cpu().numpy(),
                    depth_n[i].permute(1, 2, 0)[:, :, -1].cpu().numpy(),
                ]
            )
        # Add figures
        fig = grid_plot(image_grid)
        gray_figs = grid_plot(gray_depth_grid, img_type="gray")
        self.writer.add_figure("Validation/visualize_depths", gray_figs, epoch)
        self.writer.add_figure("Validation/visualize_image", fig, epoch)

        loss_metric.reset()
        are_metric.reset()
        log10_metric.reset()
        rmse_metric.reset()
        a1_metric.reset()
        a2_metric.reset()
        a3_metric.reset()

    def train(self, **kwargs):
        # Training parameters
        config = self.config
        optimizer = self.optimizer
        val_interval = config.system.val_interval
        writer = self.writer

        # Load data
        train_dataset = Dataset.create(
            config.train.dataset,
            split="train",
            data_root=config.system.data_root,
            transforms=self._NYU_transforms(is_train=True),
        )

        train_loader = create_loader(
            train_dataset,
            batch_size=config.train.batch_size,
            num_workers=config.system.workers,
            dryrun=config.system.dryrun,
        )

        val_dataset = Dataset.create(
            config.val.dataset,
            split="test",
            data_root=config.system.data_root,
            transforms=self._NYU_transforms(is_train=False),
        )

        val_loader = create_loader(
            val_dataset,
            batch_size=config.val.batch_size,
            num_workers=config.system.workers,
            dryrun=config.system.dryrun,
        )

        # Logging
        logger.info("Start training estimator: %s", type(self).__name__)

        self.model.to(self.device)
        n_epochs = config.train.epochs

        # Start training
        for epoch in range(1, n_epochs + 1):
            logger.info(f"Epoch[{epoch}/{n_epochs}] training started.")
            loss_metric = Loss(self._loss_fn)
            self.model.train()
            N = len(train_loader)
            accumulation_steps = self.config.train.accumulation_steps
            optimizer.zero_grad()
            for i, (image, depth) in enumerate(train_loader):
                # Prepare sample and depth
                image = image.to(self.device)
                depth_n = depth.to(self.device)

                # Predict
                output = self.model(image)

                # Compute loss
                loss = self._loss_fn(output, depth_n)

                # Backward
                loss.backward()

                if (i + 1) % accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                loss_metric.update((output, depth_n))

                # Log progress
                logger.debug(f"[{i}/{N}] Loss: {loss:.4f}")

            epoch_loss = loss_metric.compute()
            if epoch % val_interval == 0:
                self._evaluate_one_epoch(val_loader, epoch, n_epochs)

            # Record epoch's intermediate results
            writer.add_scalar("Training/Loss", epoch_loss, epoch)
            self.checkpointer.save(self, epoch=epoch)

        self.writer.close()

    def evaluate(self, **kwargs):
        config = self.config
        test_dataset = Dataset.create(
            config.test.dataset,
            split="test",
            data_root=config.system.data_root,
            transforms=self._NYU_transforms(is_train=False),
        )
        test_loader = create_loader(
            test_dataset,
            batch_size=config.test.batch_size,
            num_workers=config.system.workers,
            dryrun=config.system.dryrun,
        )

        logger.info("Start evaluating estimator: %s", type(self).__name__)
        self.model.to(self.device)
        self._evaluate_one_epoch(test_loader, 1, 1)
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
                f"Estimator loaded from {path} was trained using config: "
                f"{loaded_config}. However, the current config is: "
                f"{self.config}."
            )
