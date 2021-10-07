import json
from typing import Dict

from pycocotools.coco import COCO


def load_coco_annotations_json(annotation_file: str) -> Dict:
    f = open(annotation_file)
    data = json.load(f)
    annotations = data["annotations"]
    return annotations


def load_coco_annotations(annotation_file: str) -> COCO:
    coco = COCO(annotation_file)
    return coco
