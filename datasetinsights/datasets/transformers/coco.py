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

logger = logging.getLogger(__name__)


def uuid_to_int(input_uuid):
    try:
        u = uuid.UUID(input_uuid).int
    except (AttributeError, ValueError):
        u = int(
            hashlib.md5(str(input_uuid).encode("utf8")).hexdigest(), base=16
        )

    return u


class COCOTransformer:
    """Convert Synthetic dataset to COCO format

    Args:
        data_root (str): root directory of the dataset
        bbox2d_definition_id (str): the annotation definition id for
            2d bounding boxes in this dataset.
    """

    def __init__(self, data_root, bbox2d_definition_id):
        self._data_root = Path(data_root)
        self._bbox2d_definition_id = bbox2d_definition_id
        self._captures = Captures(
            data_root=data_root, version=const.DEFAULT_PERCEPTION_VERSION
        ).filter(def_id=bbox2d_definition_id)
        self._annotation_definitions = AnnotationDefinitions(
            data_root, version=const.DEFAULT_PERCEPTION_VERSION
        )

    def execute(self, output):
        self._copy_images(output)
        self._process_instances(output)

    def _copy_images(self, output):
        image_to_folder = Path(output) / "images"
        image_to_folder.mkdir(parents=True, exist_ok=True)
        for _, row in self._captures.iterrows():
            image_from = self._data_root / row["filename"]
            if not image_from.exists():
                continue
            image_to = image_to_folder / image_from.name
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
        for _, row in self._captures.iterrows():
            image_file = self._data_root / row["filename"]
            if not image_file.exists():
                continue
            with Image.open(image_file) as im:
                width, height = im.size
            capture_id = uuid_to_int(row["id"])
            record = {
                "license": 1,
                "file_name": Path(image_file).name,
                "coco_url": "",
                "height": height,
                "width": width,
                "date_captured": "",
                "flickr_url": "",
                "id": capture_id,
            }
            images.append(record)

        return images

    def _annotations(self):
        annotations = []
        for _, row in self._captures.iterrows():
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
        def_dict = self._annotation_definitions.get_definition(
            def_id=self._bbox2d_definition_id
        )
        categories = []
        for r in def_dict["spec"]:
            record = {
                "id": r["label_id"],
                "name": r["label_name"],
                "supercategory": "default",
            }
            categories.append(record)

        return categories
