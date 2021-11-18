import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO

from datasetinsights.io.exceptions import InvalidCOCOImageIdError


def load_img_ann_for_single_image(coco_obj: COCO, img_id: int) -> Dict:
    """
    Args:
        coco_obj (pycocotools.coco.COCO): COCO object
        img_id (int): Image id of the image
    Returns:
        Dict: Returns dict of image metadata.
    """
    try:
        img = coco_obj.loadImgs(ids=[img_id])
    except KeyError:
        raise InvalidCOCOImageIdError
    return img[0]


def load_image_from_img_ann(img_annotation: dict, data_dir: str) -> np.ndarray:
    """
    Args:
        img_annotation (dict): Image metadata dict
        data_dir (str): Directory where data(images) is located
    Returns:
        np.ndarray: Numpy array of image
    """
    image_path = os.path.join(data_dir, img_annotation["file_name"])
    img = plt.imread(image_path)
    return img


def load_annotations_for_single_img(coco_obj, img_id) -> List[Dict]:
    """
    Args:
        coco_obj (pycocotools.coco.COCO): COCO object
        img_id (int): Image id of the image
    Returns:
        List[Dict]: List of annotation objects of an image
    """
    ann_ids = coco_obj.getAnnIds(imgIds=img_id)
    annotations = coco_obj.loadAnns(ann_ids)
    return annotations
