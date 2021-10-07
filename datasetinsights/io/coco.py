import json
from typing import Dict

from pycocotools.coco import COCO


def load_coco_annotations_json(annotation_file: str) -> Dict:
    """

    Args:
        annotation_file (str): COCO annotation json file.

    Returns:
        dict: Annotations dict from the json

    """
    with open(annotation_file) as f:
        data = json.load(f)
    annotations = data["annotations"]
    return annotations


def load_coco_annotations(annotation_file: str) -> COCO:
    """

    Args:
        annotation_file (str): COCO annotation json file.

    Returns:
        pycocotools.coco.COCO: COCO object

    """
    coco = COCO(annotation_file)
    return coco
