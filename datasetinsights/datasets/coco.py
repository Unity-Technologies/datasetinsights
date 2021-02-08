import fcntl
import glob
import json
import logging
import os
import shutil
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
from .exceptions import DatasetNotFoundError

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
        coco_annotation (tuple): image and coco style dictionary.

    Returns: a tuple of image, List of BBox2D.

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
    """COCO dataset for 2D object detection.

    Before the class instantiation, it would assume that the COCO dataset is
    downloaded.

    See COCO dataset `documentation <http://cocodataset.org/#detection-2019>`_
    for more details.

    Attributes:
        root (str): root path of the data.
        transforms: callable transformation that applies to a pair of
            capture, annotation. Capture is the information captured by the
            sensor, in this case an image, and annotations, which in this
            dataset are 2d bounding box coordinates and labels.
        split (str): indicate split type of the dataset (train|val).
        label_mappings (dict): a dict of {label_id: label_name} mapping.
        coco (torchvision.datasets.CocoDetection): COCO dataset.
    """

    def __init__(
        self,
        *,
        data_path=const.DEFAULT_DATA_ROOT,
        split="train",
        transforms=None,
        remove_examples_without_boxes=True,
        **kwargs,
    ):
        """
        Args:
            data_path (str): Directory of the dataset.
            split (str): indicate split type of the dataset (train|val).
            transforms: callable transformation that applies to a pair of
            capture, annotation.
            remove_examples_without_boxes (bool): whether to remove examples
            without boxes. Defaults to True.
        """
        # todo add test split
        self.split = split
        self.root = data_path
        self._preprocess_dataset(data_path=self.root, split=self.split)
        self.coco = self._get_coco(root=self.root, image_set=split)
        if remove_examples_without_boxes:
            self.coco = _coco_remove_images_without_annotations(
                dataset=self.coco
            )
        self.transforms = transforms
        self.label_mappings = self._get_label_mappings()

    def __getitem__(self, idx) -> Tuple[Image, List[BBox2D]]:
        """
        Args:
            idx (int): index of the data.

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

    def _get_label_mappings(self):
        """get label mappings.

        Returns:
            dict: A dict containing {label_id: label_name} mappings.
        """
        ann_file_name = (
            Path(self.root) / "annotations" / f"instances_{self.split}2017.json"
        )
        label_mappings = {}
        with open(ann_file_name, "r") as ann_file:
            anns = json.load(ann_file)
            for cat in anns["categories"]:
                label_mappings[cat["id"]] = cat["name"]
        return label_mappings

    @staticmethod
    def _preprocess_dataset(data_path, split):
        """ Preprocess dataset inside data_path and un-archive if necessary.

        Args:
            data_path (str): Path where dataset is stored.
            split (str): indicate split type of the dataset (train|val).

        Return:
            Tuple: (unarchived img path, unarchived annotation path)
        """

        archive_img_file = Path(data_path) / f"{split}2017.zip"
        archive_ann_file = Path(data_path) / "annotations_trainval2017.zip"
        if archive_img_file.exists() and archive_ann_file.exists():
            unarchived_img_path = CocoDetection._unarchive_data(
                data_path, archive_img_file
            )
            unarchived_ann_path = CocoDetection._unarchive_data(
                data_path, archive_ann_file
            )
            return (unarchived_img_path, unarchived_ann_path)
        elif CocoDetection._is_dataset_files_present(data_path):
            # This is for dataset generated by unity simulation.
            return data_path
        else:
            raise DatasetNotFoundError(
                f"Expecting a file {archive_img_file} and {archive_ann_file}"
                "under {data_path}"
            )

    def _unarchive_data(self, data_path, archive_file):
        """unarchive downloaded data.
        Args:
            data_path (str): Path where dataset is stored.
            archive_file (str): archived file name.

        Returns:
            str: unarchived path.
        """
        file_descriptor = os.open(archive_file, os.O_RDONLY)
        try:
            fcntl.flock(file_descriptor, fcntl.LOCK_EX)
            unarchived_path = Path(data_path)
            if not CocoDetection._is_dataset_files_present(unarchived_path):
                shutil.unpack_archive(
                    filename=archive_file, extract_dir=unarchived_path,
                )
                logger.info(f"Unpack {archive_file} to {unarchived_path}")
        finally:
            os.close(file_descriptor)
        return unarchived_path

    @staticmethod
    def _is_dataset_files_present(data_path):
        """check whether dataset files exist.

        Args:
            data_path (str): Path where dataset is stored.

        Returns:
            bool: whether dataset files exist.
        """
        return (
            os.path.isdir(data_path)
            and any(glob.glob(f"{data_path}/*.json"))
            and any(glob.glob(f"{data_path}/*.jpg"))
        )

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
