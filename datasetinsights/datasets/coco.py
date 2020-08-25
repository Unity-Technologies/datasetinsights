import logging
import os
import zipfile
from pathlib import Path
from typing import List, Tuple

import torch
import torchvision
from PIL.Image import Image

import datasetinsights.constants as const
from datasetinsights.io.bbox import BBox2D
from datasetinsights.io.gcs import GCSClient

from .base import Dataset

ANNOTATION_FILE_TEMPLATE = "{}_{}2017.json"
COCO_GCS_PATH = "data/coco"
COCO_LOCAL_PATH = "coco"
logger = logging.getLogger(__name__)


def _coco_remove_images_without_annotations(dataset):
    """

    Args:
        dataset (torchvision.datasets.CocoDetection):

    Returns (torch.utils.data.Subset): filters dataset to exclude examples
    which either have no bounding boxes or have an invalid bounding box (a
    bounding box is invalid if it's height or width is <1).

    """

    def _has_any_empty_box(anno):
        return any(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

    def _has_valid_annotation(anno):
        # if it's empty, there is no annotation
        if len(anno) == 0:
            return False
        if _has_any_empty_box(anno):
            return False
        return True

    ids = []
    for ds_idx, img_id in enumerate(dataset.ids):
        ann_ids = dataset.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = dataset.coco.loadAnns(ann_ids)
        if _has_valid_annotation(anno):
            ids.append(ds_idx)
    dataset = torch.utils.data.Subset(dataset, ids)
    return dataset


def convert_coco2canonical(coco_annotation):
    """
    convert from a tuple of image and coco style dictionary describing the
    bboxes to a tuple of image, List of BBox2D
    Args:
        coco_annotation (tuple): image and coco style dictionary

    Returns: a tuple of image, List of BBox2D

    """
    image, targets = coco_annotation
    all_bboxes = []
    for t in targets:
        label = t["category_id"]
        bbox = t["bbox"]
        b = BBox2D(x=bbox[0], y=bbox[1], w=bbox[2], h=bbox[3], label=label)
        all_bboxes.append(b)
    return image, all_bboxes


class CocoDetection(Dataset):
    """
    http://cocodataset.org/#detection-2019
    """

    def __init__(
        self,
        *,
        data_root=const.DEFAULT_DATA_ROOT,
        split="train",
        transforms=None,
        remove_examples_without_boxes=True,
        **kwargs,
    ):
        # todo add test split
        self.split = split
        self.root = os.path.join(data_root, COCO_LOCAL_PATH)
        self.download()
        self.coco = self._get_coco(root=self.root, image_set=split)
        if remove_examples_without_boxes:
            self.coco = _coco_remove_images_without_annotations(
                dataset=self.coco
            )
        self.transforms = transforms

    def __getitem__(self, idx) -> Tuple[Image, List[BBox2D]]:
        """
        Args:
            idx:

        Returns: Image with list of bounding boxes found inside the image

        """
        image, target = convert_coco2canonical(self.coco[idx])
        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.coco)

    def _get_coco(self, root, image_set, mode="instances"):
        PATHS = {
            "train": (
                "train2017",
                os.path.join(
                    "annotations",
                    ANNOTATION_FILE_TEMPLATE.format(mode, "train"),
                ),
            ),
            "val": (
                "val2017",
                os.path.join(
                    "annotations", ANNOTATION_FILE_TEMPLATE.format(mode, "val")
                ),
            ),
        }
        img_folder, ann_file = PATHS[image_set]
        img_folder = os.path.join(root, img_folder)
        ann_file = os.path.join(root, ann_file)
        coco = torchvision.datasets.CocoDetection(img_folder, ann_file)
        return coco

    def _get_local_annotations_zip(self):
        return os.path.join(self.root, "annotations_trainval2017.zip")

    def _get_local_images_zip(self):
        return os.path.join(self.root, f"{self.split}2017.zip")

    def download(self, cloud_path=COCO_GCS_PATH):
        path = Path(self.root)
        path.mkdir(parents=True, exist_ok=True)
        client = GCSClient()
        annotations_zip_gcs = f"{cloud_path}/annotations_trainval2017.zip"
        annotations_zip_2017 = self._get_local_annotations_zip()
        logger.info(f"checking for local copy of data")
        if not os.path.exists(annotations_zip_2017):
            logger.info(f"no annotations zip file found, will download.")
            client.download(
                local_path=self.root,
                bucket=const.GCS_BUCKET,
                key=annotations_zip_gcs,
            )
            with zipfile.ZipFile(annotations_zip_2017, "r") as zip_dir:
                zip_dir.extractall(self.root)
        images_local = self._get_local_images_zip()
        images_gcs = f"{cloud_path}/{self.split}2017.zip"
        if not os.path.exists(images_local):
            logger.info(
                f"no zip file for images for {self.split} found,"
                f" will download"
            )
            client.download(
                local_path=self.root, bucket=const.GCS_BUCKET, key=images_gcs,
            )
            with zipfile.ZipFile(images_local, "r") as zip_dir:
                zip_dir.extractall(self.root)
