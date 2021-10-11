import math
import os
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from tqdm import tqdm

from datasetinsights.io.coco import load_coco_annotations
from datasetinsights.io.exceptions import (
    InvalidCOCOCategoryIdError,
    InvalidCOCOImageIdError,
)


def _load_coco_cat_data(coco_obj: COCO, cat_id: int = 1):
    try:
        data = coco_obj.loadCats(ids=cat_id)
    except KeyError:
        raise InvalidCOCOCategoryIdError
    return data[0]


def get_coco_keypoints(coco_obj: COCO, cat_id: int = 1):
    data = _load_coco_cat_data(coco_obj=coco_obj, cat_id=cat_id)
    try:
        keypoints = data["keypoints"]
    except KeyError:
        raise ValueError(f"No keypoints found for cat id: {cat_id}")
    return keypoints


def get_coco_skeleton(coco_obj: COCO, cat_id: int = 1):
    data = _load_coco_cat_data(coco_obj=coco_obj, cat_id=cat_id)
    try:
        skeleton = data["skeleton"]
    except KeyError:
        raise ValueError(f"No skeleton found for cat id: {cat_id}")
    return skeleton


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


def _get_empty_bbox_heatmap(coco_obj: COCO, cat_id: Union[List, int] = 1):
    """

    Args:
        coco_obj (pycocotools.coco.COCO): COCO object
        cat_id (Union[int, List]): List or int of category ids, Default: 1
        for person category

    Returns:
        np.ndarray: BBox numpy array of zeros

    """
    coco_img_ids = coco_obj.getImgIds(catIds=cat_id)
    max_height, max_width = 0, 0

    # calculate bbox shape according to largest image in the dataset
    for idx in coco_img_ids:
        img = load_img_ann_for_single_image(coco_obj=coco_obj, img_id=idx)
        max_height = max(img["height"], max_height)
        max_width = max(img["width"], max_width)

    bbox_heatmap = np.zeros([max_height, max_width, 1])
    return bbox_heatmap


def _get_labeled_kpt_dict(coco_obj: COCO, cat_id: int = 1):
    """

    Args:
        coco_obj (pycocotools.coco.COCO): COCO object
        cat_id (int): Category id, Default: 1 for person category

    Returns:
        Dict: Keypoint dictionary with initial values for each kp as 0.0

    """
    kpt_dict = {}
    coco_keypoints = get_coco_keypoints(coco_obj=coco_obj, cat_id=cat_id)
    for keypoint in coco_keypoints:
        kpt_dict[keypoint] = 0.0
    return kpt_dict


def _get_bbox_relative_size(bbox_area: float, image_h: float, image_w: float):
    """

    Args:
        bbox_area (float): Bounding box area size
        image_h (float): Image height
        image_w (float): Image width

    Returns:
        float: Relative size of bbox w.r.t image size

    """
    bbox_relative_size = math.sqrt(bbox_area / (image_h * image_w))
    return bbox_relative_size


def get_labeled_keypoints_dict(annotation_file: str):
    """

    Args:
        annotation_file (JSON): COCO annotations json file path

    Returns:

    """
    coco = load_coco_annotations(annotation_file=annotation_file)
    labeled_kpt_dict = _get_labeled_kpt_dict(coco_obj=coco)
    keypoints = list(labeled_kpt_dict.keys())
    img_ids = coco.getImgIds(catIds=1)
    total_instances = 0

    for idx in tqdm(img_ids):
        img_ann = load_img_ann_for_single_image(coco_obj=coco, img_id=idx)
        annotation_ids = coco.getAnnIds(
            imgIds=img_ann["id"], catIds=1, iscrowd=None
        )
        annotations = coco.loadAnns(annotation_ids)
        num_annotations = len(annotations)
        total_instances += num_annotations

        for person in range(num_annotations):

            kp_visibility_flags = annotations[person]["keypoints"][2::3]
            for i in range(len(keypoints)):
                if kp_visibility_flags[i] != 0:
                    kp_visibility_flags[keypoints[i]] += 1

    for key in labeled_kpt_dict.keys():
        labeled_kpt_dict[key] = labeled_kpt_dict[key] / total_instances

    return labeled_kpt_dict


def get_bbox_heatmap(annotation_file):
    coco = load_coco_annotations(annotation_file=annotation_file)
    img_ids = coco.getImgIds(catIds=1)
    bbox_heatmap = _get_empty_bbox_heatmap(coco_obj=coco)

    for idx in tqdm(img_ids):
        img_ann = load_img_ann_for_single_image(coco_obj=coco, img_id=idx)
        annotation_ids = coco.getAnnIds(
            imgIds=img_ann["id"], catIds=1, iscrowd=None
        )
        annotations = coco.loadAnns(annotation_ids)

        for person in range(len(annotations)):
            bbox = np.array(annotations[person]["bbox"]).astype(int)
            bbox_heatmap[
                bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2], :
            ] += 1

    return bbox_heatmap


def bbox_relative_size_list(annotation_file):
    coco = load_coco_annotations(annotation_file=annotation_file)
    img_ids = coco.getImgIds(catIds=1)
    bbox_relative_size = []

    for idx in tqdm(img_ids):
        img_ann = load_img_ann_for_single_image(coco_obj=coco, img_id=idx)
        h, w = img_ann["height"], img_ann["width"]
        annotation_ids = coco.getAnnIds(
            imgIds=img_ann["id"], catIds=1, iscrowd=None
        )
        annotations = coco.loadAnns(annotation_ids)

        for person in range(len(annotations)):
            area_size = annotations[person]["area"]
            relative_size = _get_bbox_relative_size(
                bbox_area=area_size, image_h=h, image_w=w
            )
            bbox_relative_size.append(relative_size)

    return bbox_relative_size


def _convert_coco_annotations_to_df(annotation_file):
    coco = COCO(annotation_file)
    img_ids = coco.getImgIds(catIds=1)

    coco_data = []

    for i, img_id in enumerate(img_ids):
        img_meta = load_img_ann_for_single_image(coco_obj=coco, img_id=img_id)
        ann_ids = coco.getAnnIds(imgIds=img_id)

        # basic parameters of an image
        img_file_name = img_meta["file_name"]
        w = img_meta["width"]
        h = img_meta["height"]
        # retrieve metadata for all persons in the current image
        meta = coco.loadAnns(ann_ids)

        # iterate over all metadata
        for m in meta:
            coco_data.append(
                {
                    "image_id": m["image_id"],
                    "path": img_file_name,
                    "width": int(w),
                    "height": int(h),
                    "is_crowd": m["iscrowd"],
                    "bbox": m["bbox"],
                    "area": m["area"],
                    "num_keypoints": m["num_keypoints"],
                    "keypoints": m["keypoints"],
                }
            )
    # create dataframe with image paths
    coco_df = pd.DataFrame(coco_data)
    coco_df.set_index("image_id", inplace=True)

    return coco_df


def _get_annotations_per_img(annotation_file):
    coco_df = _convert_coco_annotations_to_df(annotation_file=annotation_file)
    annotated_persons_df = coco_df[(coco_df["is_crowd"] == 0)]

    persons_in_img_df = pd.DataFrame(
        {"cnt": annotated_persons_df[["path"]].value_counts()}
    )
    persons_in_img_df.reset_index(level=[0], inplace=True)

    # group by counter so we will get the dataframe with number of annotated
    # people in a single image
    persons_in_img_cnt_df = persons_in_img_df.groupby(["cnt"]).count()

    # extract arrays
    x_occurrences = persons_in_img_cnt_df.index.values
    y_images = persons_in_img_cnt_df["path"].values

    return x_occurrences, y_images


def get_bbox_per_img_dict(annotation_file):
    x_occ, y_img = _get_annotations_per_img(annotation_file=annotation_file)
    bbox_dict = {}
    for i in range(1, max(x_occ) + 1):
        if i in x_occ:
            bbox_dict[i] = y_img[i - 1] / sum(y_img)
        else:
            bbox_dict[i] = 0

    return bbox_dict


def _get_keypoints_per_img(annotation_file):
    coco_df = _convert_coco_annotations_to_df(annotation_file=annotation_file)
    annotated_persons_df = coco_df[(coco_df["is_crowd"] == 0)]

    kp_in_bbox = pd.DataFrame(
        {"cnt": annotated_persons_df[["num_keypoints"]].value_counts()}
    )
    kp_in_bbox.reset_index(level=[0], inplace=True)
    kp_in_bbox.sort_values(by=["num_keypoints"], inplace=True)

    # extract arrays
    num_kp = kp_in_bbox["num_keypoints"].values
    count = kp_in_bbox["cnt"].values

    return num_kp, count


def get_keypoints_per_bbox_dict(annotation_file):
    num_kp, count = _get_keypoints_per_img(annotation_file=annotation_file)
    kpt_dict = {}
    for i in range(0, max(num_kp) + 1):
        if i in num_kp:
            kpt_dict[i] = count[i - 1] / sum(count)
        else:
            kpt_dict[i] = 0

    return kpt_dict
