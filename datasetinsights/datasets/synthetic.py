""" Simulation Dataset Catalog
"""


import logging

from pyquaternion import Quaternion

from datasetinsights.io.bbox import BBox2D, BBox3D

logger = logging.getLogger(__name__)


def read_bounding_box_3d(annotation, label_mappings=None):
    """ Convert dictionary representations of 3d bounding boxes into objects
    of the BBox3d class

    Args:
        annotation (List[dict]): 3D bounding box annotation
        label_mappings (dict): a dict of {label_id: label_name} mapping

    Returns:
        A list of 3d bounding box objects
    """

    bboxes = []

    for b in annotation:
        label_id = b["label_id"]
        translation = b["translation"]
        translation = [translation["x"], translation["y"], translation["z"]]
        size = b["size"]
        size = [size["x"], size["y"], size["z"]]
        rotation = b["rotation"]
        rotation = Quaternion(
            b=rotation["x"], c=rotation["y"], d=rotation["z"], a=rotation["w"]
        )

        if label_mappings and label_id not in label_mappings:
            continue
        box = BBox3D(
            translation=translation,
            size=size,
            label=label_id,
            sample_token=0,
            score=1,
            rotation=rotation,
        )
        bboxes.append(box)

    return bboxes


def read_bounding_box_2d(annotation, label_mappings=None):
    """Convert dictionary representations of 2d bounding boxes into objects
    of the BBox2D class

    Args:
        annotation (List[dict]): 2D bounding box annotation
        label_mappings (dict): a dict of {label_id: label_name} mapping

    Returns:
        A list of 2D bounding box objects
    """
    bboxes = []
    for b in annotation:
        label_id = b["label_id"]
        x = b["x"]
        y = b["y"]
        w = b["width"]
        h = b["height"]
        if label_mappings and label_id not in label_mappings:
            continue
        box = BBox2D(label=label_id, x=x, y=y, w=w, h=h)
        bboxes.append(box)

    return bboxes
