import hashlib
import json
import logging
import shutil
import uuid
from pathlib import Path

from PIL import Image

import datasetinsights.constants as const
from datasetinsights.datasets.unity_perception import (
    AnnotationDefinitions,
    Captures,
)
from datasetinsights.datasets.unity_perception.validation import NoRecordError

logger = logging.getLogger(__name__)


def uuid_to_int(input_uuid):
    try:
        u = uuid.UUID(input_uuid).int
    except (AttributeError, ValueError):
        u = int(
            hashlib.md5(str(input_uuid).encode("utf8")).hexdigest(), base=16
        )

    return u


class COCOInstancesTransformer:
    """Convert Synthetic dataset to COCO format.

    This transformer convert Synthetic dataset into annotations in instance
    format (e.g. instances_train2017.json, instances_val2017.json)

    Note: We assume "valid images" in the COCO dataset must contain at least one
    bounding box annotation. Therefore, all images that contain no bounding
    boxes will be dropped. Instance segmentation are considered optional
    in the converted dataset as some synthetic dataset might be generated
    without it.

    Args:
        data_root (str): root directory of the dataset
    """

    # The annotation_definition.name is not a reliable way to know the type
    # of annotation definition. This will be improved once the perception
    # package introduced the annotation definition type in the future.
    BBOX_NAME = r"^(?:2[dD]\s)?bounding\sbox$"
    INSTANCE_SEGMENTATION_NAME = r"^instance\ssegmentation$"

    def __init__(self, data_root):
        self._data_root = Path(data_root)

        ann_def = AnnotationDefinitions(
            data_root, version=const.DEFAULT_PERCEPTION_VERSION
        )
        self._bbox_def = ann_def.find_by_name(self.BBOX_NAME)
        try:
            self._instance_segmentation_def = ann_def.find_by_name(
                self.INSTANCE_SEGMENTATION_NAME
            )
        except NoRecordError as e:
            logger.warning(
                "Can't find instance segmentation annotations in the dataset. "
                "The converted file will not contain instance segmentation."
            )
            logger.warning(e)
            self._instance_segmentation_def = None

        captures = Captures(
            data_root=data_root, version=const.DEFAULT_PERCEPTION_VERSION
        )
        self._bbox_captures = captures.filter(self._bbox_def["id"])
        if self._instance_segmentation_def:
            self._instance_segmentation_captures = captures.filter(
                self._instance_segmentation_def
            )

    def execute(self, output):
        """Execute COCO Transformer

        Args:
            output (str): the output directory where converted dataset will
              be stored.
        """
        self._copy_images(output)
        self._process_instances(output)

    def _copy_images(self, output):
        image_to_folder = Path(output) / "images"
        image_to_folder.mkdir(parents=True, exist_ok=True)
        for _, row in self._bbox_captures.iterrows():
            image_from = self._data_root / row["filename"]
            if not image_from.exists():
                continue
            capture_id = uuid_to_int(row["id"])
            image_to = image_to_folder / f"camera_{capture_id}.png"
            shutil.copy(str(image_from), str(image_to))

    def _process_instances(self, output):
        output = Path(output) / "annotations"
        output.mkdir(parents=True, exist_ok=True)
        instances = {
            "info": {"description": "COCO compatible Synthetic Dataset"},
            "licences": [{"url": "", "id": 1, "name": "default"}],
            "images": self._images(),
            "annotations": self._annotations(),
            "categories": self._categories(),
        }
        output_file = output / "instances.json"
        with open(output_file, "w") as out:
            json.dump(instances, out)

    def _images(self):
        images = []
        for _, row in self._bbox_captures.iterrows():
            image_file = self._data_root / row["filename"]
            if not image_file.exists():
                continue
            with Image.open(image_file) as im:
                width, height = im.size
            capture_id = uuid_to_int(row["id"])
            record = {
                "file_name": f"camera_{capture_id}.png",
                "height": height,
                "width": width,
                "id": capture_id,
            }
            images.append(record)

        return images

    def _annotations(self):
        annotations = []
        for _, row in self._bbox_captures.iterrows():
            image_id = uuid_to_int(row["id"])
            for ann in row["annotation.values"]:
                x = ann["x"]
                y = ann["y"]
                w = ann["width"]
                h = ann["height"]
                area = float(w) * float(h)
                record = {
                    "segmentation": [],  # TODO: parse instance segmentation map
                    "area": area,
                    "iscrowd": 0,
                    "image_id": image_id,
                    "bbox": [x, y, w, h],
                    "category_id": ann["label_id"],
                    "id": uuid_to_int(row["annotation.id"])
                    | uuid_to_int(ann["instance_id"]),
                }
                annotations.append(record)

        return annotations

    def _categories(self):
        categories = []
        for r in self._bbox_def["spec"]:
            record = {
                "id": r["label_id"],
                "name": r["label_name"],
                "supercategory": "default",
            }
            categories.append(record)

        return categories
