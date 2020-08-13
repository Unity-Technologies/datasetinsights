import glob
import logging
import math
import random
import time
from os.path import join
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as FF
from torch.autograd import Variable
from torch.optim.lr_scheduler import _LRScheduler

from datasetinsights.datasets import Dataset
from datasetinsights.io.bbox import BBox2D
from datasetinsights.io.loader import create_loader
from datasetinsights.io.transforms import (
    BlurImage,
    CenterCrop,
    GrayImage,
    ScaleBBox,
    ShiftBBox,
)

from .base import Estimator


logger = logging.getLogger(__name__)


def corner2center(corner):
    """
    Converts a corner representation to center representation
    x1,y1,x2,y2 -> cx,cy,w,h
    :param corner: Corner or np.array 4*N
    :return: Center or 4 np.array N
    """
    x1, y1, x2, y2 = corner[0], corner[1], corner[2], corner[3]
    x = (x1 + x2) * 0.5
    y = (y1 + y2) * 0.5
    w = x2 - x1
    h = y2 - y1
    return x, y, w, h


def center2corner(center):
    """
    Converts a corner representation to center representation
    cx,cy,w,h -> x1,y1,x2,y2
    :param center: Center or np.array 4*N
    :return: Corner or np.array 4*N
    """
    x, y, w, h = center[0], center[1], center[2], center[3]
    x1 = x - w * 0.5
    y1 = y - h * 0.5
    x2 = x + w * 0.5
    y2 = y + h * 0.5
    return x1, y1, x2, y2


def anyform2canonicalBBox(form, rep_type):
    """
    Helper function for form conversion
    Converting any form corner to center or opposite
    """
    if rep_type == "corner":
        x1, y1, x2, y2 = form[0], form[1], form[2], form[3]
        w, h = x2 - x1, y2 - y1

    elif rep_type == "center":
        corner = center2corner(form)
        x1, y1, x2, y2 = corner[0], corner[1], corner[2], corner[3]
        w, h = x2 - x1, y2 - y1

    return BBox2D(label=None, x=x1, y=y1, w=w, h=h)


def toBBox(image, shape, template_size=127):
    """
    Converting coordinates to standard BBox implementation
    """
    imh, imw = image.shape[:2]
    if len(shape) == 4:
        w, h = shape[2] - shape[0], shape[3] - shape[1]
    else:
        w, h = shape
    context_amount = 0.5
    exemplar_size = template_size
    wc_z = w + context_amount * (w + h)
    hc_z = h + context_amount * (w + h)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = exemplar_size / s_z
    w = w * scale_z
    h = h * scale_z
    cx, cy = imw // 2, imh // 2
    bbox = anyform2canonicalBBox([cx, cy, w, h], "center")
    return bbox


def crop_hwc(image, bbox, out_sz, padding=(0, 0, 0)):
    """
    Cropping image given a bounding box
    Args: Image and modified bounding box
    Returns: Cropped Image
    """
    x1, y1, w, h = bbox.x, bbox.y, bbox.w, bbox.h
    bbox = [x1, y1, x1 + w, y1 + h]
    bbox = [float(x) for x in bbox]
    a = (out_sz - 1) / (bbox[2] - bbox[0])
    b = (out_sz - 1) / (bbox[3] - bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c], [0, b, d]]).astype(np.float)
    crop = cv2.warpAffine(
        image,
        mapping,
        (out_sz, out_sz),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=padding,
    )
    return crop


def draw(image, box, name):
    """
        Args: image, bbox
        Saves an image with bbox drawn on it
        Only for debugging-> Not used anywhere
    """
    if box is not None:
        box = [box.x, box.y, box.x + box.w, box.y + box.h]
        box = [round(x) for x in box]
    else:
        box = [0, 0, 0, 0]
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    image = image.copy()
    # https://github.com/opencv/opencv/issues/14866#issuecomment-504632481 weird
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0))
    cv2.imwrite(name, image)


"""
    Generates a random number between 2 given numbers
"""


def genarate_random_number(a, b):
    return a + (random.random() * (b - a))


"""
Helper functions sent to the transforms class
"""

helper = {
    "corner2center": corner2center,
    "center2corner": center2corner,
    "random_number": genarate_random_number,
}


def get_transforms(center_crop, blur, gray, shift, scale, resize, flip):

    """
    # Should maintain this order
    # (x) TODO: center crop both template and search -buggy
    # make bounding box on both
    # scale
    # shift
    # Any order below this
    # grayscale
    # blur
    # resize - not tested
    # flip - not tested
    """
    transforms = []
    if center_crop[0]:
        transforms.append(CenterCrop(size=center_crop[1]))
    if scale[0]:
        transforms.append(ScaleBBox(helper=helper, scale=scale[1]))
    if shift[0]:
        transforms.append(ShiftBBox(helper=helper, shift=shift[1]))
    if gray[0]:
        transforms.append(GrayImage(p=gray[1]))
    if blur[0]:
        transforms.append(BlurImage(p=blur[1]))

    return SiamMaskCompose(transforms)


class SiamMaskCompose:
    def __init__(self, transforms):
        self.transforms = transforms
        # This is a number to calculate offset
        # Offset improves training performance
        self.rgbVar = np.array(
            [
                [-0.55919361, 0.98062831, -0.41940627],
                [1.72091413, 0.19879334, -1.82968581],
                [4.64467907, 4.73710203, 4.88324118],
            ],
            dtype=np.float32,
        )

    def __call__(self, image, annotations, size, mask):

        """
        Applies the transforms populated in the transforms list
        """

        # For debugging - actual bbox of object
        bbox_orig = toBBox(image, annotations)

        # Storing shape before it changes
        shape = image.shape

        # Converts the template/search image annotations into bbox canonical
        bbox = bbox_orig

        # generate center crop in canonical bbox and scale/shift
        crop_bbox = anyform2canonicalBBox(
            [shape[0] // 2, shape[1] // 2, size - 1, size - 1], "center"
        )
        crop_bbox = self.transforms[1](crop_bbox, shape)
        crop_bbox = anyform2canonicalBBox(crop_bbox, "corner")

        crop_bbox = self.transforms[2](crop_bbox, shape)
        crop_bbox = anyform2canonicalBBox(crop_bbox, "corner")

        # Adjust image according to the new scaled/shifted bounding boxes
        x1 = crop_bbox.x
        y1 = crop_bbox.y
        bbox = anyform2canonicalBBox(
            [
                bbox.x - x1,
                bbox.y - y1,
                bbox.x + bbox.w - x1,
                bbox.y + bbox.h - y1,
            ],
            "corner",
        )
        scale_x, scale_y = (
            self.transforms[1].scale_x,
            self.transforms[1].scale_y,
        )
        bbox = anyform2canonicalBBox(
            [
                bbox.x / scale_x,
                bbox.y / scale_y,
                (bbox.x + bbox.w) / scale_x,
                (bbox.y + bbox.h) / scale_y,
            ],
            "corner",
        )
        image = crop_hwc(image, crop_bbox, size)
        if mask is not None:
            mask = crop_hwc(mask, crop_bbox, size)
        # Smooth image with removing offset
        # this is supposed improve training performance*
        offset = np.dot(self.rgbVar, np.random.randn(3, 1))
        offset = offset[::-1]  # bgr 2 rgb
        offset = offset.reshape(3)
        image = image - offset

        # Apply the rest of the transforms
        for t in self.transforms[3:]:
            image, _ = t((image, None))

        return image, bbox, mask, bbox_orig


class Anchors:
    def __init__(self, cfg):
        """
            Args: config file
            Initializes the anchor box specs
        """
        self.stride = 8
        self.ratios = [0.33, 0.5, 1, 2, 3]
        self.scales = [8]
        self.round_dight = 0
        self.image_center = 0
        self.size = 0
        self.anchor_density = 1

        self.__dict__.update(cfg)

        self.anchor_num = (
            len(self.scales) * len(self.ratios) * (self.anchor_density ** 2)
        )
        self.anchors = None  # in single position (anchor_num*4)
        self.all_anchors = None  # in all position 2*(4*anchor_num*h*w)
        self.generate_anchors()

    def generate_anchors(self):
        """
            Generates the anchor boxes with spec in config file
        """
        self.anchors = np.zeros((self.anchor_num, 4), dtype=np.float32)

        size = self.stride * self.stride
        count = 0
        anchors_offset = self.stride / self.anchor_density
        anchors_offset = np.arange(self.anchor_density) * anchors_offset
        anchors_offset = anchors_offset - np.mean(anchors_offset)
        x_offsets, y_offsets = np.meshgrid(anchors_offset, anchors_offset)

        for x_offset, y_offset in zip(x_offsets.flatten(), y_offsets.flatten()):
            for r in self.ratios:
                if self.round_dight > 0:
                    ws = round(math.sqrt(size * 1.0 / r), self.round_dight)
                    hs = round(ws * r, self.round_dight)
                else:
                    ws = int(math.sqrt(size * 1.0 / r))
                    hs = int(ws * r)

                for s in self.scales:
                    w = ws * s
                    h = hs * s
                    self.anchors[count][:] = [
                        -w * 0.5 + x_offset,
                        -h * 0.5 + y_offset,
                        w * 0.5 + x_offset,
                        h * 0.5 + y_offset,
                    ][:]
                    count += 1

    def generate_all_anchors(self, im_c, size):
        """
            Helper function for generating anchors
        """
        if self.image_center == im_c and self.size == size:
            return False
        self.image_center = im_c
        self.size = size

        a0x = im_c - size // 2 * self.stride
        ori = np.array([a0x] * 4, dtype=np.float32)
        zero_anchors = self.anchors + ori

        x1 = zero_anchors[:, 0]
        y1 = zero_anchors[:, 1]
        x2 = zero_anchors[:, 2]
        y2 = zero_anchors[:, 3]

        x1, y1, x2, y2 = map(
            lambda x: x.reshape(self.anchor_num, 1, 1), [x1, y1, x2, y2]
        )
        cx, cy, w, h = corner2center([x1, y1, x2, y2])

        disp_x = np.arange(0, size).reshape(1, 1, -1) * self.stride
        disp_y = np.arange(0, size).reshape(1, -1, 1) * self.stride

        cx = cx + disp_x
        cy = cy + disp_y

        # broadcast
        zero = np.zeros((self.anchor_num, size, size), dtype=np.float32)
        cx, cy, w, h = map(lambda x: x + zero, [cx, cy, w, h])
        x1, y1, x2, y2 = center2corner([cx, cy, w, h])

        self.all_anchors = np.stack([x1, y1, x2, y2]), np.stack([cx, cy, w, h])
        return True


class AnchorTargetLayer:
    def __init__(self):
        """
            self.thr_high -> Above this anchor box is positive
            self.thr_low = 0.3 -> Below this anchor box is negative
            self.negative = 16 -> Number of boxes to select
            self.rpn_batch = 16 - > Batch Size
            self.positive = 16 -> Number of boxes to select
        """
        self.thr_high = 0.6
        self.thr_low = 0.3
        self.negative = 16
        self.rpn_batch = 16
        self.positive = 16

    def __call__(self, anchor, target, size, neg=False, need_iou=False):
        anchor_num = anchor.anchors.shape[0]

        cls = np.zeros((anchor_num, size, size), dtype=np.int64)
        cls[...] = -1  # -1 ignore 0 negative 1 positive
        delta = np.zeros((4, anchor_num, size, size), dtype=np.float32)
        delta_weight = np.zeros((anchor_num, size, size), dtype=np.float32)

        # Function to randomly select the number of +ve/-ve boxes
        def select(position, keep_num=16):
            num = position[0].shape[0]
            if num <= keep_num:
                return position, num
            slt = np.arange(num)
            np.random.shuffle(slt)
            slt = slt[:keep_num]
            return tuple(p[slt] for p in position), keep_num

        if neg:
            left = size // 2 - 3
            r = size // 2 + 3 + 1

            cls[:, left:r, left:r] = 0

            neg, neg_num = select(np.where(cls == 0), self.negative)
            cls[:] = -1
            cls[neg] = 0

            if not need_iou:
                return cls, delta, delta_weight
            else:
                overlap = np.zeros((anchor_num, size, size), dtype=np.float32)
                return cls, delta, delta_weight, overlap

        tcx, tcy, tw, th = corner2center(
            [target.x, target.y, target.x + target.w, target.y + target.h]
        )
        anchor_box = anchor.all_anchors[0]
        anchor_center = anchor.all_anchors[1]
        x1, y1, x2, y2 = (
            anchor_box[0],
            anchor_box[1],
            anchor_box[2],
            anchor_box[3],
        )
        cx, cy, w, h = (
            anchor_center[0],
            anchor_center[1],
            anchor_center[2],
            anchor_center[3],
        )

        # delta (basically difference of imagebbox and anchorbox)
        delta[0] = (tcx - cx) / w
        delta[1] = (tcy - cy) / h
        delta[2] = np.log(tw / w)
        delta[3] = np.log(th / h)

        def IoU(rect1, rect2):
            # overlap
            x1, y1, x2, y2 = rect1[0], rect1[1], rect1[2], rect1[3]
            tx1, ty1, tx2, ty2 = rect2[0], rect2[1], rect2[2], rect2[3]

            xx1 = np.maximum(tx1, x1)
            yy1 = np.maximum(ty1, y1)
            xx2 = np.minimum(tx2, x2)
            yy2 = np.minimum(ty2, y2)

            ww = np.maximum(0, xx2 - xx1)
            hh = np.maximum(0, yy2 - yy1)

            area = (x2 - x1) * (y2 - y1)

            target_a = (tx2 - tx1) * (ty2 - ty1)

            inter = ww * hh
            overlap = inter / (area + target_a - inter)

            return overlap

        # IoU
        overlap = IoU(
            [x1, y1, x2, y2],
            [target.x, target.y, target.x + target.w, target.y + target.h],
        )
        # Canonical BBox class does not support multiple IoUs together,
        # so using the IoU from authors
        # overlap = anyform2canonicalBBox([x1,y1,x2,y2],
        # rep_type = "corner").iou(anyform2canonicalBBox
        # (target,rep_type = "corner"))
        # print("Overlap: ",overlap)

        pos = np.where(overlap > self.thr_high)
        neg = np.where(overlap < self.thr_low)

        pos, pos_num = select(pos, self.positive)
        neg, neg_num = select(neg, self.rpn_batch - pos_num)

        cls[pos] = 1
        delta_weight[pos] = 1.0 / (pos_num + 1e-6)

        cls[neg] = 0

        if not need_iou:
            return cls, delta, delta_weight
        else:
            return cls, delta, delta_weight, overlap


class SiamMask(Estimator):
    """
        Main SiamMask estimator class
    """

    def __init__(
        self,
        *,
        config,
        writer,
        checkpointer,
        device,
        gpu=0,
        rank=0,
        train_mode=True,
        **kwargs,
    ):

        # config files should contain the hyperparameters of the model
        # TB Writer object comes from the main class itself
        # checkpointer should be used to save progress after each iteration
        # Device = cpu/cuda
        # GPU = gpu id
        # rank = should be 0 for serial (dont change this)
        # kwargs are for additional params (add as we go ahead)
        logger.info(f"Initializing SiamMask")
        self.config = config
        self.device = device
        self.writer = writer
        self.gpu = gpu
        self.rank = rank
        # model_name = f"siammask_{self.config.backbone}"
        logger.info(f"gpu: {self.gpu}, rank: {self.rank}")

        self.train_mode = train_mode

        # Declare Model
        if self.config.model_type == "3branch":
            self.model = SiamMask3Branch(
                anchor_cfg=config["train"]["anchors"], device=device
            ).to(self.device)

        # Declare Checkpoint
        self.checkpointer = checkpointer

        self.model.to(self.device)

    def build_scheduler(self, optimizer, config, epochs=50, last_epoch=-1):
        """
            Builds the lr scheduler according to the configfile
        """
        warmup_epoch = config["warmup"]["epoch"]
        sc1 = Scheduler(
            optimizer,
            config["warmup"]["start_lr"],
            config["warmup"]["end_lr"],
            warmup_epoch,
            last_epoch,
        )
        sc2 = Scheduler(
            optimizer,
            config["start_lr"],
            config["end_lr"],
            epochs - warmup_epoch,
            last_epoch,
        )
        return WarmupScheduler(optimizer, sc1, sc2, epochs, last_epoch)

    def get_opt_lrsched(self, model, config, epoch):
        """
            Builds optimizer using lr correspinding to epoch
        """
        backbone_feature = model.features.param_groups(config["start_lr"])
        if len(backbone_feature) == 0:
            trainable_params = model.rpn_model.param_groups(
                config["start_lr"], key="mask"
            )
        else:
            trainable_params = (
                backbone_feature
                + model.rpn_model.param_groups(config["start_lr"])
                + model.mask_model.param_groups(config["start_lr"])
            )

        optimizer = torch.optim.SGD(
            trainable_params,
            config["warmup"]["start_lr"],
            momentum=config["momentum"],
            weight_decay=config["weight_decay"],
        )

        lr_scheduler = self.build_scheduler(
            optimizer, config, epochs=config["epoch"]
        )

        lr_scheduler.step(epoch)

        return optimizer, lr_scheduler

    def train_loop(
        self, train_loader, val_loader, config, writer, checkpointer, model
    ):

        logger.info("Training starting...")
        path = Path(config["model_save_path"])
        path.mkdir(parents=True, exist_ok=True)

        batches_per_epoch = (
            len(train_loader.dataset) // config["epoch"] // config["batch_size"]
        )
        epoch = 0
        # start_epoch = if we wanna resume training
        # Declare Average collector:
        avg = AverageMeter()
        # Start Timer:
        end = time.time()
        epoch = 0
        optimizer, lr_scheduler = self.get_opt_lrsched(
            model.module, config, epoch
        )

        cur_lr = lr_scheduler.get_cur_lr()

        for iter, input in enumerate(train_loader):
            # If epoch and iterations seem out of order, restart from 0
            if epoch != (iter // batches_per_epoch):
                epoch = iter // batches_per_epoch

                # Every time the epoch changes save the best model
                self.save(
                    {
                        "epoch": epoch,  # ,'state_dict': model.state_dict(),
                        "state_dict": model.module.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    join(path, "checkpoint_e%d.pth" % (epoch)),
                )

                if epoch == config["epoch"]:
                    return

                # Unfreeze more layers as a % of epochs trained
                if self.model.module.features.unfix(epoch / config["epoch"]):
                    logger.info("unfix part model.")
                    optimizer, lr_scheduler = self.get_opt_lrsched(
                        model.module, config, epoch
                    )

                lr_scheduler.step(epoch)
                cur_lr = lr_scheduler.get_cur_lr()

                logger.info("epoch:{}".format(epoch))

            tb_index = iter
            if iter % batches_per_epoch == 0 and iter != 0:
                for idx, pg in enumerate(optimizer.param_groups):
                    logger.info("epoch {} lr {}".format(epoch, pg["lr"]))
                    writer.add_scalar(
                        "lr/group%d" % (idx + 1), pg["lr"], tb_index
                    )

            data_time = time.time() - end
            avg.update(data_time=data_time)

            x = {
                "template": torch.autograd.Variable(input[0]).to(self.device),
                "search": torch.autograd.Variable(input[1]).to(self.device),
                "label_cls": torch.autograd.Variable(input[2]).to(self.device),
                "label_loc": torch.autograd.Variable(input[3]).to(self.device),
                "label_loc_weight": torch.autograd.Variable(input[4]).to(
                    self.device
                ),
                "label_mask": torch.autograd.Variable(input[6]).to(self.device),
                "label_mask_weight": torch.autograd.Variable(input[7]).to(
                    self.device
                ),
            }

            outputs = self.model(x)

            rpn_cls_loss, rpn_loc_loss, rpn_mask_loss = (
                torch.mean(outputs["losses"][0]),
                torch.mean(outputs["losses"][1]),
                torch.mean(outputs["losses"][2]),
            )
            mask_iou_mean, mask_iou_at_5, mask_iou_at_7 = (
                torch.mean(outputs["accuracy"][0]),
                torch.mean(outputs["accuracy"][1]),
                torch.mean(outputs["accuracy"][2]),
            )

            cls_weight, reg_weight, mask_weight = (
                config["loss_weights"]["feature"],
                config["loss_weights"]["rpn"],
                config["loss_weights"]["mask"],
            )

            loss = (
                rpn_cls_loss * cls_weight
                + rpn_loc_loss * reg_weight
                + rpn_mask_loss * mask_weight
            )

            optimizer.zero_grad()
            loss.backward()

            # Clip gradients of all 3 losses
            if config["clip"]:
                torch.nn.utils.clip_grad_norm_(
                    model.module.features.parameters(),
                    config["clip"]["feature"],
                )
                torch.nn.utils.clip_grad_norm_(
                    model.module.rpn_model.parameters(), config["clip"]["rpn"]
                )
                torch.nn.utils.clip_grad_norm_(
                    model.module.mask_model.parameters(), config["clip"]["mask"]
                )
            if (
                not math.isnan(loss.item())
                or not math.isinf(loss.item())
                or not loss.item() > 1e4
            ):
                optimizer.step()
            else:
                logger.info("Loss corrupted! Stopping training")
                return

            siammask_loss = loss.item()

            batch_time = time.time() - end

            avg.update(
                batch_time=batch_time,
                rpn_cls_loss=rpn_cls_loss,
                rpn_loc_loss=rpn_loc_loss,
                rpn_mask_loss=rpn_mask_loss,
                siammask_loss=siammask_loss,
                mask_iou_mean=mask_iou_mean,
                mask_iou_at_5=mask_iou_at_5,
                mask_iou_at_7=mask_iou_at_7,
            )

            writer.add_scalar("loss/cls", rpn_cls_loss, tb_index)
            writer.add_scalar("loss/loc", rpn_loc_loss, tb_index)
            writer.add_scalar("loss/mask", rpn_mask_loss, tb_index)
            writer.add_scalar("mask/mIoU", mask_iou_mean, tb_index)
            writer.add_scalar("mask/AP_.5", mask_iou_at_5, tb_index)
            writer.add_scalar("mask/AP_.7", mask_iou_at_7, tb_index)

            end = time.time()

            if (iter + 1) % 1 == 0:
                logger.info(
                    "Epoch: [{0}][{1}/{2}]\n \
                    lr: {lr:.6f}\n{batch_time:s}\n{data_time:s}"
                    "\n{rpn_cls_loss:s}\
                    \n{rpn_loc_loss:s}\
                    \n{rpn_mask_loss:s}\
                    \n{siammask_loss:s}"
                    "\n{mask_iou_mean:s}\
                    \n{mask_iou_at_5:s}\
                    \n{mask_iou_at_7:s}".format(
                        epoch + 1,
                        (iter + 1) % batches_per_epoch,
                        batches_per_epoch,
                        lr=cur_lr,
                        batch_time=avg.batch_time,
                        data_time=avg.data_time,
                        rpn_cls_loss=avg.rpn_cls_loss,
                        rpn_loc_loss=avg.rpn_loc_loss,
                        rpn_mask_loss=avg.rpn_mask_loss,
                        siammask_loss=avg.siammask_loss,
                        mask_iou_mean=avg.mask_iou_mean,
                        mask_iou_at_5=avg.mask_iou_at_5,
                        mask_iou_at_7=avg.mask_iou_at_7,
                    )
                )
            print(cur_lr)

    def save(self, state, filename="checkpoint.pth"):
        torch.save(state, filename)

    def train(self):
        self.train_mode = True
        # Parallelize model
        self.model = torch.nn.DataParallel(self.model, device_ids=[self.gpu])
        train_dataset = Dataset.create(
            name=self.config["train"]["dataset"]["name"][0],
            anchors=Anchors(self.config["train"]),
            anchor_target=AnchorTargetLayer(),
            config=self.config["train"],
            transforms=get_transforms,
        )

        train_dataset.split = "train"
        # Create dataloaders
        train_loader = create_loader(
            train_dataset, batch_size=self.config["train"]["batch_size"]
        )
        # TODO: create val dataloader
        # val_dataset = SiamDataset(config["val"])
        # val_dataset.split = "val"
        # val_dataset.create(config["val"])
        # val_loader = create_loader(val_dataset)
        val_loader = None
        logger.info(f"Dataloaders created!")

        self.train_loop(
            train_loader=train_loader,
            val_loader=val_loader,
            config=self.config["train"],
            writer=self.writer,
            checkpointer=self.checkpointer,
            model=self.model,
        )

    def evaluate(self):
        self.train_mode = False
        load_path = self.config["test"]["dataset"]["saved_model"][0]
        print(load_path)
        model = load_pretrain(self.model, load_path)
        model.eval()
        model = model.to(self.device)

        test_dataset = {}
        datapath = self.config["test"]["dataset"]["dataroot"][0]
        dataannot = self.config["test"]["dataset"]["annot_file"][0]

        with open(dataannot) as f:
            videos = [v.strip() for v in f.readlines()]
        for video in videos:
            test_dataset[video] = {}
            test_dataset[video]["anno_files"] = sorted(
                glob.glob(join(datapath, "Annotations/480p", video, "*.png"))
            )
            test_dataset[video]["image_files"] = sorted(
                glob.glob(join(datapath, "JPEGImages/480p", video, "*.jpg"))
            )
            test_dataset[video]["name"] = video

        logger.info(f"Dataset DAVIS loaded (using 2016)")
        model = load_pretrain(
            self.model, self.config["test"]["dataset"]["saved_model"][0]
        )

        iou_lists = []
        speed_list = []

        for v_id, video in enumerate(test_dataset.keys(), start=1):

            iou_list, speed = self.track_vos(
                model, test_dataset[video], device=self.device, v_id=v_id
            )
            iou_lists.append(iou_list)

        thrs = np.arange(0.3, 0.5, 0.05)

        for thr, iou in zip(thrs, np.mean(np.concatenate(iou_lists), axis=0)):
            logger.info(
                "Segmentation Threshold {:.2f} mIoU: {:.3f}".format(thr, iou)
            )

        logger.info("Mean Speed: {:.2f} FPS".format(np.mean(speed_list)))

    def track_vos(self, model, video, device, v_id):
        """
            Args: model, 1 single video, gpu id
            Returns: IOU value
        """
        image_files = video["image_files"]
        from PIL import Image

        annos = [np.array(Image.open(x)) for x in video["anno_files"]]
        annos_init = [annos[0]]
        annos = [(anno > 0).astype(np.uint8) for anno in annos]
        annos_init = [
            (anno_init > 0).astype(np.uint8) for anno_init in annos_init
        ]
        object_ids = [o_id for o_id in np.unique(annos[0]) if o_id != 0]
        if len(object_ids) != len(annos_init):
            annos_init = annos_init * len(object_ids)
        object_num = len(
            object_ids
        )  # Always set as 1 because we are not doing multiple object tracking
        toc = 0
        pred_masks = (
            np.zeros(
                (
                    object_num,
                    len(image_files),
                    annos[0].shape[0],
                    annos[0].shape[1],
                )
            )
            - 1
        )
        for obj_id, o_id in enumerate(object_ids):
            start_frame, end_frame = 0, len(image_files)
            for f, image_file in enumerate(image_files):
                im = cv2.imread(image_file)
                tic = cv2.getTickCount()
                if f == start_frame:  # init
                    mask = annos_init[obj_id] == o_id
                    x, y, w, h = cv2.boundingRect((mask).astype(np.uint8))
                    cx, cy = x + w / 2, y + h / 2
                    target_pos = np.array([cx, cy])
                    target_sz = np.array([w, h])
                    state = self.siamese_init(
                        im, target_pos, target_sz, model, device=self.device
                    )  # init tracker
                    # print(state)
                    # exit(0)
                elif end_frame >= f > start_frame:  # tracking
                    state = self.siamese_track(
                        state, im, mask_enable=True
                    )  # track
                    mask = state["mask"]
                    toc += cv2.getTickCount() - tic
                if end_frame >= f >= start_frame:
                    pred_masks[obj_id, f, :, :] = mask
        toc /= cv2.getTickFrequency()

        thrs = np.arange(0.3, 0.5, 0.05)

        if len(annos) == len(image_files):
            multi_mean_iou = self.MultiBatchIouMeter(
                thrs,
                pred_masks,
                annos,
                start=video["start_frame"] if "start_frame" in video else None,
                end=video["end_frame"] if "end_frame" in video else None,
            )
            for i in range(object_num):
                for j, thr in enumerate(thrs):
                    logger.info(
                        "Multi Object{:20s} IOU at {:.2f}: {:.4f}".format(
                            video["name"] + "_" + str(i + 1),
                            thr,
                            multi_mean_iou[i, j],
                        )
                    )
        else:
            multi_mean_iou = []

        visualization = True
        if visualization:
            pred_mask_final = np.array(pred_masks)
            pred_mask_final = (
                np.argmax(pred_mask_final, axis=0).astype("uint8") + 1
            ) * (
                np.max(pred_mask_final, axis=0)
                > self.config["test"]["dataset"]["confidence"]
            ).astype(
                "uint8"
            )
            COLORS = np.random.randint(
                128, 255, size=(object_num, 3), dtype="uint8"
            )
            COLORS = np.vstack([[0, 0, 0], COLORS]).astype("uint8")
            mask = COLORS[pred_mask_final]
            save_path = self.config["test"]["dataset"]["save_vis"][0]
            Path(save_path).mkdir(parents=True, exist_ok=True)
            save_path_vid = Path(save_path + str(v_id))
            save_path_vid.mkdir(parents=True, exist_ok=True)
            for f, image_file in enumerate(image_files):
                output = (
                    (0.4 * cv2.imread(image_file)) + (0.6 * mask[f, :, :, :])
                ).astype("uint8")
                svp = join(save_path_vid, (str(f) + ".jpg"))
                cv2.imwrite(str(svp), output)

        logger.info(
            "({:d}) Video: {:12s} Time: {:02.1f}s Speed: {:3.1f}fps".format(
                v_id, video["name"], toc, f * len(object_ids) / toc
            )
        )

        return multi_mean_iou, f * len(object_ids) / toc

    def generate_anchor(self, cfg, score_size):
        """
            Generates anchors during testing
        """
        anchors = Anchors(cfg)
        anchor = anchors.anchors
        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        anchor = np.stack(
            [(x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1], 1
        )

        total_stride = anchors.stride
        anchor_num = anchor.shape[0]

        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = -(score_size // 2) * total_stride
        xx, yy = np.meshgrid(
            [ori + total_stride * dx for dx in range(score_size)],
            [ori + total_stride * dy for dy in range(score_size)],
        )
        xx, yy = (
            np.tile(xx.flatten(), (anchor_num, 1)).flatten(),
            np.tile(yy.flatten(), (anchor_num, 1)).flatten(),
        )
        anchor[:, 0], anchor[:, 1] = (
            xx.astype(np.float32),
            yy.astype(np.float32),
        )
        return anchor

    def get_subwindow_tracking(
        self, im, pos, model_sz, original_sz, avg_chans, out_mode="torch"
    ):
        """
            Generates box around mask
        """
        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_sz
        im_sz = im.shape
        c = (original_sz + 1) / 2
        context_xmin = round(pos[0] - c)
        context_xmax = context_xmin + sz - 1
        context_ymin = round(pos[1] - c)
        context_ymax = context_ymin + sz - 1
        left_pad = int(max(0.0, -context_xmin))
        top_pad = int(max(0.0, -context_ymin))
        right_pad = int(max(0.0, context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0.0, context_ymax - im_sz[0] + 1))

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad

        # zzp: a more easy speed version
        r, c, k = im.shape
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            te_im = np.zeros(
                (r + top_pad + bottom_pad, c + left_pad + right_pad, k),
                np.uint8,
            )
            te_im[top_pad : top_pad + r, left_pad : left_pad + c, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad : left_pad + c, :] = avg_chans
            if bottom_pad:
                te_im[r + top_pad :, left_pad : left_pad + c, :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, c + left_pad :, :] = avg_chans
            im_patch_original = te_im[
                int(context_ymin) : int(context_ymax + 1),
                int(context_xmin) : int(context_xmax + 1),
                :,
            ]
        else:
            im_patch_original = im[
                int(context_ymin) : int(context_ymax + 1),
                int(context_xmin) : int(context_xmax + 1),
                :,
            ]

        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))
        else:
            im_patch = im_patch_original
        # cv2.imshow('crop', im_patch)
        # cv2.waitKey(0)
        if out_mode in "torch":
            img = np.transpose(im_patch, (2, 0, 1))
            return torch.from_numpy(img).float()
        else:
            return im_patch

    def siamese_init(self, im, target_pos, target_sz, model, device="cpu"):
        """
            Helper function to perform tracking.
            Returns: information from one video frame to the next
        """
        state = dict()
        state["im_h"] = im.shape[0]
        state["im_w"] = im.shape[1]

        exemplar_size = 127
        instance_size = 255
        total_stride = 8
        base_size = 8
        # Dimensions of the score map
        score_size = (
            (instance_size - exemplar_size) // total_stride + 1 + base_size
        )
        # maximum score in the classification branch -> per-pixel sigmoid
        # binarise the output of the mask branch at context_amount
        context_amount = 0.5

        net = model
        anchor_num = model.anchor_num
        avg_chans = np.mean(im, axis=(0, 1))

        wc_z = target_sz[0] + context_amount * sum(target_sz)
        hc_z = target_sz[1] + context_amount * sum(target_sz)
        s_z = round(np.sqrt(wc_z * hc_z))
        # initialize the exemplar
        z_crop = self.get_subwindow_tracking(
            im, target_pos, exemplar_size, s_z, avg_chans
        )

        z = Variable(z_crop.unsqueeze(0))
        net.template(z.to(self.device))

        window = np.outer(np.hanning(score_size), np.hanning(score_size))

        window = np.tile(window.flatten(), anchor_num)

        state["net"] = net
        state["avg_chans"] = avg_chans
        state["window"] = window
        state["target_pos"] = target_pos
        state["target_sz"] = target_sz
        return state

    def siamese_track(self, state, im, mask_enable=True):
        """
            Helper function to perform appropriate tracking
        """
        net = state["net"]
        avg_chans = state["avg_chans"]
        window = state["window"]
        target_pos = state["target_pos"]
        target_sz = state["target_sz"]

        penalty_k = 0.09
        window_influence = 0.39
        lr = 0.38
        seg_thr = 0.3  # for mask
        exemplar_size = 127  # input z size
        instance_size = 255  # input x size (search region)
        total_stride = 8
        out_size = 63  # for mask
        base_size = 8
        score_size = (
            (instance_size - exemplar_size) // total_stride + 1 + base_size
        )
        context_amount = 0.5

        wc_x = target_sz[1] + context_amount * sum(target_sz)
        hc_x = target_sz[0] + context_amount * sum(target_sz)
        s_x = np.sqrt(wc_x * hc_x)
        scale_x = exemplar_size / s_x
        d_search = (instance_size - exemplar_size) / 2
        pad = d_search / scale_x
        s_x = s_x + 2 * pad
        crop_box = [
            target_pos[0] - round(s_x) / 2,
            target_pos[1] - round(s_x) / 2,
            round(s_x),
            round(s_x),
        ]

        # extract scaled crops for search region x at previous target position
        x_crop = Variable(
            self.get_subwindow_tracking(
                im, target_pos, instance_size, round(s_x), avg_chans
            ).unsqueeze(0)
        )

        score, delta, mask = net.track_mask(x_crop.to(self.device))

        delta = (
            delta.permute(1, 2, 3, 0)
            .contiguous()
            .view(4, -1)
            .data.cpu()
            .numpy()
        )
        score = (
            FF.softmax(
                score.permute(1, 2, 3, 0)
                .contiguous()
                .view(2, -1)
                .permute(1, 0),
                dim=1,
            )
            .data[:, 1]
            .cpu()
            .numpy()
        )

        anchor = self.generate_anchor(net.anchors, score_size)
        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]

        def change(r):
            return np.maximum(r, 1.0 / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            sz2 = (w + pad) * (h + pad)
            return np.sqrt(sz2)

        def sz_wh(wh):
            pad = (wh[0] + wh[1]) * 0.5
            sz2 = (wh[0] + pad) * (wh[1] + pad)
            return np.sqrt(sz2)

        # size penalty
        target_sz_in_crop = target_sz * scale_x
        s_c = change(
            sz(delta[2, :], delta[3, :]) / (sz_wh(target_sz_in_crop))
        )  # scale penalty
        r_c = change(
            (target_sz_in_crop[0] / target_sz_in_crop[1])
            / (delta[2, :] / delta[3, :])
        )  # ratio penalty

        penalty = np.exp(-(r_c * s_c - 1) * penalty_k)
        pscore = penalty * score

        # cos window (motion model)
        pscore = pscore * (1 - window_influence) + window * window_influence
        best_pscore_id = np.argmax(pscore)

        pred_in_crop = delta[:, best_pscore_id] / scale_x
        lr = penalty[best_pscore_id] * score[best_pscore_id] * lr  # lr for OTB

        res_x = pred_in_crop[0] + target_pos[0]
        res_y = pred_in_crop[1] + target_pos[1]

        res_w = target_sz[0] * (1 - lr) + pred_in_crop[2] * lr
        res_h = target_sz[1] * (1 - lr) + pred_in_crop[3] * lr

        target_pos = np.array([res_x, res_y])
        target_sz = np.array([res_w, res_h])

        # for Mask Branch
        # if mask_enable:
        best_pscore_id_mask = np.unravel_index(
            best_pscore_id, (5, score_size, score_size)
        )
        delta_x, delta_y = best_pscore_id_mask[2], best_pscore_id_mask[1]

        mask = (
            mask[0, :, delta_y, delta_x]
            .sigmoid()
            .squeeze()
            .view(out_size, out_size)
            .cpu()
            .data.numpy()
        )

        def crop_back(image, bbox, out_sz, padding=-1):
            a = (out_sz[0] - 1) / bbox[2]
            b = (out_sz[1] - 1) / bbox[3]
            c = -a * bbox[0]
            d = -b * bbox[1]
            mapping = np.array([[a, 0, c], [0, b, d]]).astype(np.float)
            crop = cv2.warpAffine(
                image,
                mapping,
                (out_sz[0], out_sz[1]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=padding,
            )
            return crop

        s = crop_box[2] / instance_size
        sub_box = [
            crop_box[0] + (delta_x - base_size / 2) * total_stride * s,
            crop_box[1] + (delta_y - base_size / 2) * total_stride * s,
            s * exemplar_size,
            s * exemplar_size,
        ]
        s = out_size / sub_box[2]
        back_box = [
            -sub_box[0] * s,
            -sub_box[1] * s,
            state["im_w"] * s,
            state["im_h"] * s,
        ]
        mask_in_img = crop_back(mask, back_box, (state["im_w"], state["im_h"]))

        target_mask = (mask_in_img > seg_thr).astype(np.uint8)
        if cv2.__version__[-5] == "4":
            contours, _ = cv2.findContours(
                target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
        else:
            _, contours, _ = cv2.findContours(
                target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
        cnt_area = [cv2.contourArea(cnt) for cnt in contours]
        if len(contours) != 0 and np.max(cnt_area) > 100:
            contour = contours[np.argmax(cnt_area)]  # use max area polygon
            polygon = contour.reshape(-1, 2)
            # pbox = cv2.boundingRect(polygon)  # Min Max Rectangle
            prbox = cv2.boxPoints(cv2.minAreaRect(polygon))  # Rotated Rectangle

            # box_in_img = pbox
            rbox_in_img = prbox

        target_pos[0] = max(0, min(state["im_w"], target_pos[0]))
        target_pos[1] = max(0, min(state["im_h"], target_pos[1]))
        target_sz[0] = max(10, min(state["im_w"], target_sz[0]))
        target_sz[1] = max(10, min(state["im_h"], target_sz[1]))

        state["target_pos"] = target_pos
        state["target_sz"] = target_sz
        state["score"] = score[best_pscore_id]
        state["mask"] = mask_in_img if mask_enable else []
        state["ploygon"] = rbox_in_img if mask_enable else []
        return state

    def MultiBatchIouMeter(self, thrs, outputs, targets, start=None, end=None):
        targets = np.array(targets)
        outputs = np.array(outputs)

        num_frame = targets.shape[0]
        if start is None:
            object_ids = np.array(list(range(outputs.shape[0]))) + 1
        else:
            object_ids = [int(id) for id in start]

        num_object = len(object_ids)
        res = np.zeros((num_object, len(thrs)), dtype=np.float32)

        output_max_id = np.argmax(outputs, axis=0).astype("uint8") + 1
        outputs_max = np.max(outputs, axis=0)
        for k, thr in enumerate(thrs):
            output_thr = outputs_max > thr
            for j in range(num_object):
                target_j = targets == object_ids[j]

                if start is None:
                    start_frame, end_frame = 1, num_frame - 1
                else:
                    start_frame, end_frame = (
                        start[str(object_ids[j])] + 1,
                        end[str(object_ids[j])] - 1,
                    )
                iou = []
                for i in range(start_frame, end_frame):
                    pred = (output_thr[i] * output_max_id[i]) == (j + 1)
                    mask_sum = (pred == 1).astype(np.uint8) + (
                        target_j[i] > 0
                    ).astype(np.uint8)
                    intxn = np.sum(mask_sum == 2)
                    union = np.sum(mask_sum > 0)
                    if union > 0:
                        iou.append(intxn / union)
                    elif union == 0 and intxn == 0:
                        iou.append(1)
                res[j, k] = np.mean(iou)
        return res


class WarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup, normal, epochs=50, last_epoch=-1):
        # optimizer, sc1,     sc2,   epochs,    last_epoch
        self.epochs = epochs

        normal = normal.lr_spaces
        warmup = warmup.lr_spaces  # [::-1]
        self.lr_spaces = np.concatenate([warmup, normal])
        self.start_lr = normal[0]

        super(WarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_cur_lr(self):
        return self.lr_spaces[self.last_epoch]

    def get_lr(self):
        epoch = self.last_epoch
        return [
            self.lr_spaces[epoch] * pg["initial_lr"] / self.start_lr
            for pg in self.optimizer.param_groups
        ]


class Scheduler(_LRScheduler):
    def __init__(
        self, optimizer, start_lr=0.03, end_lr=5e-4, epochs=50, last_epoch=-1
    ):
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.epochs = epochs
        self.lr_spaces = np.logspace(
            math.log10(start_lr), math.log10(end_lr), epochs
        )
        super(Scheduler, self).__init__(optimizer, last_epoch)

    def get_cur_lr(self):
        return self.lr_spaces[self.last_epoch]

    def get_lr(self):
        epoch = self.last_epoch
        return [
            self.lr_spaces[epoch] * pg["initial_lr"] / self.start_lr
            for pg in self.optimizer.param_groups
        ]

    def __repr__(self):
        return "({}) lr spaces: \n{}".format(
            self.__class__.__name__, self.lr_spaces
        )


# Contains all the main functions for training
class SiamMaskBase(nn.Module):
    def __init__(self, anchors=None, o_sz=63, g_sz=127):
        super(SiamMaskBase, self).__init__()
        self.anchors = anchors  # anchor_cfg
        self.anchor_num = len(self.anchors["ratios"]) * len(
            self.anchors["scales"]
        )
        self.anchor = Anchors(anchors)
        self.features = None
        self.rpn_model = None
        self.mask_model = None
        self.o_sz = o_sz
        self.g_sz = g_sz
        self.upSample = nn.UpsamplingBilinear2d(size=[g_sz, g_sz])

        self.all_anchors = None
        self.device = None
        self.training = True

    def set_all_anchors(self, image_center, size):
        # cx,cy,w,h
        if not self.anchor.generate_all_anchors(image_center, size):
            return
        all_anchors = self.anchor.all_anchors[1]  # cx, cy, w, h
        self.all_anchors = torch.from_numpy(all_anchors).float().cuda()
        self.all_anchors = [self.all_anchors[i] for i in range(4)]

    def feature_extractor(self, x):
        return self.features(x)

    def rpn(self, template, search):
        pred_cls, pred_loc = self.rpn_model(template, search)
        return pred_cls, pred_loc

    def mask(self, template, search):
        pred_mask = self.mask_model(template, search)
        return pred_mask

    def _add_rpn_loss(
        self,
        label_cls,
        label_loc,
        lable_loc_weight,
        label_mask,
        label_mask_weight,
        rpn_pred_cls,
        rpn_pred_loc,
        rpn_pred_mask,
    ):
        rpn_loss_cls = self.select_cross_entropy_loss(rpn_pred_cls, label_cls)
        rpn_loss_loc = self.weight_l1_loss(
            rpn_pred_loc, label_loc, lable_loc_weight
        )
        rpn_loss_mask, iou_m, iou_5, iou_7 = self.select_mask_logistic_loss(
            rpn_pred_mask, label_mask, label_mask_weight
        )
        return rpn_loss_cls, rpn_loss_loc, rpn_loss_mask, iou_m, iou_5, iou_7

    def run(self, template, search, softmax=False):
        """
        run network
        """
        template_feature = self.feature_extractor(template)
        search_feature = self.feature_extractor(search)
        rpn_pred_cls, rpn_pred_loc = self.rpn(template_feature, search_feature)
        rpn_pred_mask = self.mask(
            template_feature, search_feature
        )  # (b, 63*63, w, h)

        if softmax:
            rpn_pred_cls = self.softmax(rpn_pred_cls)
        return (
            rpn_pred_cls,
            rpn_pred_loc,
            rpn_pred_mask,
            template_feature,
            search_feature,
        )

    def softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2 // 2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = FF.log_softmax(cls, dim=4)
        return cls

    def forward(self, input):
        """
        :param input: dict of input with keys of:
                'template': [b, 3, h1, w1], input template image.
                'search': [b, 3, h2, w2], input search image.
                'label_cls':[b, max_num_gts, 5] or None(self.training==False),
                                     each gt contains x1,y1,x2,y2,class.
        :return: dict of loss, predict, accuracy
        """
        template = input["template"]
        search = input["search"]
        if self.training:
            label_cls = input["label_cls"]
            label_loc = input["label_loc"]
            lable_loc_weight = input["label_loc_weight"]
            label_mask = input["label_mask"]
            label_mask_weight = input["label_mask_weight"]

        (
            rpn_pred_cls,
            rpn_pred_loc,
            rpn_pred_mask,
            template_feature,
            search_feature,
        ) = self.run(template, search, softmax=self.training)

        outputs = dict()

        outputs["predict"] = [
            rpn_pred_loc,
            rpn_pred_cls,
            rpn_pred_mask,
            template_feature,
            search_feature,
        ]

        if self.training:
            (
                rpn_loss_cls,
                rpn_loss_loc,
                rpn_loss_mask,
                iou_acc_mean,
                iou_acc_5,
                iou_acc_7,
            ) = self._add_rpn_loss(
                label_cls,
                label_loc,
                lable_loc_weight,
                label_mask,
                label_mask_weight,
                rpn_pred_cls,
                rpn_pred_loc,
                rpn_pred_mask,
            )
            outputs["losses"] = [rpn_loss_cls, rpn_loss_loc, rpn_loss_mask]
            outputs["accuracy"] = [iou_acc_mean, iou_acc_5, iou_acc_7]

        return outputs

    def template(self, z):
        self.zf = self.feature_extractor(z)
        cls_kernel, loc_kernel = self.rpn_model.template(self.zf)
        return cls_kernel, loc_kernel

    def track(self, x, cls_kernel=None, loc_kernel=None, softmax=False):
        xf = self.feature_extractor(x)
        rpn_pred_cls, rpn_pred_loc = self.rpn_model.track(
            xf, cls_kernel, loc_kernel
        )
        if softmax:
            rpn_pred_cls = self.softmax(rpn_pred_cls)
        return rpn_pred_cls, rpn_pred_loc

    def get_cls_loss(self, pred, label, select):
        if select.nelement() == 0:
            return pred.sum() * 0.0
        pred = torch.index_select(pred, 0, select)
        label = torch.index_select(label, 0, select)

        return FF.nll_loss(pred, label)

    def select_cross_entropy_loss(self, pred, label):
        pred = pred.view(-1, 2)
        label = label.view(-1)
        pos = Variable(label.data.eq(1).nonzero().squeeze()).to(self.device)
        neg = Variable(label.data.eq(0).nonzero().squeeze()).to(self.device)

        loss_pos = self.get_cls_loss(pred, label, pos)
        loss_neg = self.get_cls_loss(pred, label, neg)
        return loss_pos * 0.5 + loss_neg * 0.5

    def weight_l1_loss(self, pred_loc, label_loc, loss_weight):
        """
        :param pred_loc: [b, 4k, h, w]
        :param label_loc: [b, 4k, h, w]
        :param loss_weight:  [b, k, h, w]
        :return: loc loss value
        """
        # print(pred_loc.size(),label_loc.size())
        b, _, sh, sw = pred_loc.size()
        pred_loc = pred_loc.view(b, 4, -1, sh, sw)
        diff = (pred_loc - label_loc).abs()
        diff = diff.sum(dim=1).view(b, -1, sh, sw)
        loss = diff * loss_weight
        return loss.sum().div(b)

    def select_mask_logistic_loss(self, p_m, mask, weight, o_sz=63, g_sz=127):
        weight = weight.view(-1)
        pos = Variable(weight.data.eq(1).nonzero().squeeze())
        if pos.nelement() == 0:
            return p_m.sum() * 0, p_m.sum() * 0, p_m.sum() * 0, p_m.sum() * 0

        p_m = p_m.permute(0, 2, 3, 1).contiguous().view(-1, 1, o_sz, o_sz)
        p_m = torch.index_select(p_m, 0, pos)
        p_m = nn.UpsamplingBilinear2d(size=[g_sz, g_sz])(p_m)
        p_m = p_m.view(-1, g_sz * g_sz)

        mask_uf = FF.unfold(mask, (g_sz, g_sz), padding=32, stride=8)
        mask_uf = (
            torch.transpose(mask_uf, 1, 2).contiguous().view(-1, g_sz * g_sz)
        )

        mask_uf = torch.index_select(mask_uf, 0, pos)
        loss = FF.soft_margin_loss(p_m, mask_uf)
        iou_m, iou_5, iou_7 = self.iou_measure(p_m, mask_uf)
        return loss, iou_m, iou_5, iou_7

    def iou_measure(self, pred, label):
        pred = pred.ge(0).type(torch.uint8)
        mask_sum = torch.add(
            pred.eq(1).type(torch.uint8), (label.eq(1).type(torch.uint8))
        )
        intxn = torch.sum(mask_sum == 2, dim=1).float()
        union = torch.sum(mask_sum > 0, dim=1).float()
        iou = intxn / union

        return (
            torch.mean(iou),
            (torch.sum(iou > 0.5).float() / iou.shape[0]),
            (torch.sum(iou > 0.7).float() / iou.shape[0]),
        )


# Helper Class and functions for cross correlation step
# i.e. correlating(convolving) search with template as a kernel
class DepthCorr(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3):
        super(DepthCorr, self).__init__()
        # adjust layer for asymmetrical features
        self.conv_kernel = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.conv_search = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, kernel_size=1),
        )

    def forward_corr(self, kernel, input):
        kernel = self.conv_kernel(kernel)
        input = self.conv_search(input)
        feature = self.conv2d_dw_group(input, kernel)
        return feature

    def forward(self, kernel, search):
        feature = self.forward_corr(kernel, search)
        out = self.head(feature)
        return out

    def conv2d_dw_group(self, x, kernel):
        batch, channel = kernel.shape[:2]
        x = x.view(
            1, batch * channel, x.size(2), x.size(3)
        )  # 1 * (b*c) * k * k
        kernel = kernel.view(
            batch * channel, 1, kernel.size(2), kernel.size(3)
        )  # (b*c) * 1 * H * W
        out = FF.conv2d(x, kernel, groups=batch * channel)
        out = out.view(batch, channel, out.size(2), out.size(3))
        return out


# Resnet Helper classes and functions
"""
    Helper Functions for declaring Resnet - resnet50
    Classes for building the Resnet - ResNet, Bottleneck
    Classes for building siamese network - ResDownS, ResDown
    Helper for loading pre-trained weights into Resnet:
        f_lambda
        remove_prefix
        check_keys
        load_pretrain

"""
# Helper functions to read the pre-trained weights to the Resnet Model


"""
    Helper function to remove the word "module"
"""


def f_lambda(x, prefix):
    if x.startswith(prefix):
        return x.split(prefix, 1)[-1]
    else:
        return x


def remove_prefix(state_dict, prefix):
    """ Model is stored with parameters share common prefix 'module.' """
    logger.info("remove prefix '{}'".format(prefix))
    return {f_lambda(key, prefix): value for key, value in state_dict.items()}


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    if len(missing_keys) > 0:
        logger.info("[Warning] missing keys: {}".format(missing_keys))
        logger.info("missing keys:{}".format(len(missing_keys)))
    if len(unused_pretrained_keys) > 0:
        logger.info(
            "[Warning] unused_pretrained_keys: {}".format(
                unused_pretrained_keys
            )
        )
        logger.info(
            "unused checkpoint keys:{}".format(len(unused_pretrained_keys))
        )
    logger.info("used keys:{}".format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, "load NONE from pretrained checkpoint"
    return True


def load_pretrain(model, pretrained_path):
    """
        Args : Model path
        Loads a pre-trained resnet model to init the siamese net
    """
    logger.info("load pretrained model from {}".format(pretrained_path))
    if not torch.cuda.is_available():
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage
        )
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(
            pretrained_path,
            map_location=lambda storage, loc: storage.cuda(device),
        )

    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(
            pretrained_dict["state_dict"], "module."
        )
    else:
        pretrained_dict = remove_prefix(pretrained_dict, "module.")

    try:
        check_keys(model, pretrained_dict)
    except AssertionError:
        logger.info(
            '[Warning]: using pretrain as features. \
            Adding "features." as prefix'
        )
        new_dict = {}
        for k, v in pretrained_dict.items():
            k = "features." + k
            new_dict[k] = v
        pretrained_dict = new_dict
        check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


# Parent class conataining helper functions for
# freezing weights of the layers at any point of training
class MultiStageFeature(nn.Module):
    def __init__(self):
        super(MultiStageFeature, self).__init__()
        self.feature_size = -1
        self.layers = []
        self.train_num = -1
        # Will be overriden from child class ideally
        # This basically says, reading from right to left:
        # when model is 0.5 trained increase trainable layers to 3
        # change_point [0, 0.5]
        # train_nums[1, 3]
        self.change_point = []
        self.train_nums = []

    def forward():
        raise NotImplementedError

    # This ratio is the percentage of training completed
    # (current_epoch/total_epochs)
    def unfix(self, ratio=0.0):
        if self.train_num == -1:
            self.train_num = 0
            self.unlock()
            self.eval()
        for p, t in reversed(list(zip(self.change_point, self.train_nums))):
            if ratio >= p:
                if self.train_num != t:
                    self.train_num = t
                    self.unlock()
                    return True
                break
        return False

    # Set which layers will be trainable
    def train_layers(self):
        return self.layers[: self.train_num]

    def unlock(self):
        # Make all grads False then only set trainable ones True
        for p in self.parameters():
            p.requires_grad = False

        logger.info("Current training {} layers:\n\t".format(self.train_num))
        for m in self.train_layers():
            for p in m.parameters():
                p.requires_grad = True

    def train(self, mode):
        self.training = mode
        if not mode:
            super(MultiStageFeature, self).train(False)
        else:
            for m in self.train_layers():
                m.train(True)

        return self

    def load_model(self, f="pretrain.model"):
        with open(f) as f:
            pretrained_dict = torch.load(f)
            model_dict = self.state_dict()
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items() if k in model_dict
            }
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)


# ResDown* are the classes for declaring the Siamese network part of the code

# Main ResNet class for backbone of Siamese network - Shared
class ResNet(nn.Module):
    def __init__(self, block, layers, layer4=False, layer3=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=0, bias=False  # 3
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2
        )  # 31x31, 15x15

        self.feature_size = 128 * block.expansion

        if layer3:
            self.layer3 = self._make_layer(
                block, 256, layers[2], stride=1, dilation=2
            )  # 15x15, 7x7
            self.feature_size = (256 + 128) * block.expansion
        else:
            self.layer3 = lambda x: x  # identity

        if layer4:
            self.layer4 = self._make_layer(
                block, 512, layers[3], stride=1, dilation=4
            )  # 7x7, 3x3
            self.feature_size = 512 * block.expansion
        else:
            self.layer4 = lambda x: x  # identity

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        dd = dilation
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride == 1 and dilation == 1:
                downsample = nn.Sequential(
                    nn.Conv2d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else:
                if dilation > 1:
                    dd = dilation // 2
                    padding = dd
                else:
                    dd = 1
                    padding = 0
                downsample = nn.Sequential(
                    nn.Conv2d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=3,
                        stride=stride,
                        bias=False,
                        padding=padding,
                        dilation=dd,
                    ),
                    nn.BatchNorm2d(planes * block.expansion),
                )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, dilation=dd)
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print x.size()
        x = self.maxpool(x)
        # print x.size()

        p1 = self.layer1(x)
        p2 = self.layer2(p1)
        p3 = self.layer3(p2)
        # p3 = torch.cat([p2, p3], 1)

        # log_once("p3 {}".format(p3.size()))
        p4 = self.layer4(p3)

        return p2, p3, p4


# Bottleneck class - part of Resnet
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # padding = (2 - stride) + (dilation // 2 - 1)
        padding = 2 - stride
        assert (
            stride == 1 or dilation == 1
        ), "stride and dilation must have one equals to zero at least"
        if dilation > 1:
            padding = dilation
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=padding,
            bias=False,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if out.size() != residual.size():
            print(out.size(), residual.size())
        out += residual

        out = self.relu(out)

        return out

    def param_groups(self, start_lr, feature_mult=1):
        params = filter(lambda x: x.requires_grad, self.parameters())
        params = [{"params": params, "lr": start_lr * feature_mult}]
        return params


# Helper function to declare our custom Resnet backbone
def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    # if pretrained:
    #     model.load_state_dict(
    #         model_zoo.load_url(
    #             "https://download.pytorch.org/models/resnet50-19c8e357.pth"
    #         )
    #     )
    return model


# Adjust layers (unshared)
class ResDownS(nn.Module):
    def __init__(self, inplane, outplane):
        super(ResDownS, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(inplane, outplane, kernel_size=1, bias=False),
            nn.BatchNorm2d(outplane),
        )

    def forward(self, x):
        x = self.downsample(x)
        if x.size(3) < 20:
            _l = 4
            r = -4
            x = x[:, :, _l:r, _l:r]
        return x


# Siamese network (shared ResNet)+(unshared ResDownS)
class ResDown(MultiStageFeature):
    def __init__(self, pretrain=False):
        super(ResDown, self).__init__()
        # features is the original Siamese network
        # -> till layer3 of resnet 50 (shared)
        self.features = resnet50(layer3=True, layer4=False)
        # Initialized the Resnet by the weights
        if pretrain:
            load_pretrain(self.features, "resnet.model")
        # This is the adjust layer
        # which is not shared by template and search images (unshared)
        self.downsample = ResDownS(1024, 256)

        self.layers = [
            self.downsample,
            self.features.layer2,
            self.features.layer3,
        ]
        self.train_nums = [1, 3]
        self.change_point = [0, 0.5]

        # This controls when, what layers will be frozen
        # As this is init, so all layers will be trainable
        # if unfix(0.5) is passed, the number of layers will
        # be frozen depicted by train_nums and change_point
        self.unfix(0.0)

    def param_groups(self, start_lr, feature_mult=1):
        lr = start_lr * feature_mult

        def _params(module, mult=1):
            params = list(
                filter(lambda x: x.requires_grad, module.parameters())
            )
            if len(params):
                return [{"params": params, "lr": lr * mult}]
            else:
                return []

        groups = []
        groups += _params(self.downsample)
        groups += _params(self.features, 0.1)
        return groups

    def forward(self, x):
        output = self.features(x)
        # Output is of the form : (layer2, layer3, layer4)
        # As we dont need layer4 or layer2, we take output[1]
        p3 = self.downsample(output[1])
        # p3 is basically the expected feature block.
        # template_p3 will be used to cross correlate with search_p3
        return p3


# Building RPN model (for 3 branch)
class UP(nn.Module):
    def __init__(self, anchor_num=5, feature_in=256, feature_out=256):
        super(UP, self).__init__()

        self.anchor_num = anchor_num
        self.feature_in = feature_in
        self.feature_out = feature_out

        self.cls_output = 2 * self.anchor_num
        self.loc_output = 4 * self.anchor_num

        self.cls = DepthCorr(feature_in, feature_out, self.cls_output)
        self.loc = DepthCorr(feature_in, feature_out, self.loc_output)

    def forward(self, z_f, x_f):
        cls = self.cls(z_f, x_f)
        loc = self.loc(z_f, x_f)
        return cls, loc

    def param_groups(self, start_lr, feature_mult=1, key=None):
        if key is None:
            params = filter(lambda x: x.requires_grad, self.parameters())
        else:
            params = [
                v
                for k, v in self.named_parameters()
                if (key in k) and v.requires_grad
            ]
        params = [{"params": params, "lr": start_lr * feature_mult}]
        return params


# Class for segmntation branch of the model (for both 2 and 3 branch)
class MaskCorr(nn.Module):
    def __init__(self, oSz=63):
        super(MaskCorr, self).__init__()
        self.oSz = oSz
        self.mask = DepthCorr(256, 256, self.oSz ** 2)

    def forward(self, z, x):
        return self.mask(z, x)

    def param_groups(self, start_lr, feature_mult=1):
        params = filter(lambda x: x.requires_grad, self.parameters())
        params = [{"params": params, "lr": start_lr * feature_mult}]
        return params


# Building the complete model - SiamMaskBase is the main parent class
# 3Branch version - Mask+RPN, 2Branch - Mask+Score Map
class SiamMask3Branch(SiamMaskBase):
    def __init__(self, anchor_cfg, device=None):
        super(SiamMask3Branch, self).__init__(anchors=anchor_cfg)
        self.features = ResDown(pretrain=True)
        # For the 3 branch version the rpn model is used
        self.rpn_model = UP(
            anchor_num=self.anchor_num, feature_in=256, feature_out=256
        )
        self.mask_model = MaskCorr()
        logger.info(f"3 Branch Model declared!")
        self.device = device

    def template(self, template):
        self.zf = self.features(template)

    def track(self, search):
        search = self.features(search)
        rpn_pred_cls, rpn_pred_loc = self.rpn(self.zf, search)
        return rpn_pred_cls, rpn_pred_loc

    def track_mask(self, search):
        search = self.features(search)
        rpn_pred_cls, rpn_pred_loc = self.rpn(self.zf, search)
        pred_mask = self.mask(self.zf, search)
        return rpn_pred_cls, rpn_pred_loc, pred_mask


"""
    Helper classes for storing average of tracked metrics
"""


class Meter(object):
    def __init__(self, name, val, avg):
        self.name = name
        self.val = val
        self.avg = avg

    def __repr__(self):
        return "{name}: {val:.6f} ({avg:.6f})".format(
            name=self.name, val=self.val, avg=self.avg
        )

    def __format__(self, *tuples, **kwargs):
        return self.__repr__()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = {}
        self.sum = {}
        self.count = {}

    def update(self, batch=1, **kwargs):
        val = {}
        for k in kwargs:
            val[k] = kwargs[k] / float(batch)
        self.val.update(val)
        for k in kwargs:
            if k not in self.sum:
                self.sum[k] = 0
                self.count[k] = 0
            self.sum[k] += kwargs[k]
            self.count[k] += batch

    def __repr__(self):
        s = ""
        for k in self.sum:
            s += self.format_str(k)
        return s

    def format_str(self, attr):
        return "{name}: {val:.6f} ({avg:.6f}) ".format(
            name=attr,
            val=float(self.val[attr]),
            avg=float(self.sum[attr]) / self.count[attr],
        )

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return super(AverageMeter, self).__getattr__(attr)
        if attr not in self.sum:
            print("invalid key '{}'".format(attr))
            return Meter(attr, 0, 0)
        return Meter(attr, self.val[attr], self.avg(attr))

    def avg(self, attr):
        return float(self.sum[attr]) / self.count[attr]
