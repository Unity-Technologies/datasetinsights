import os
import random
from typing import List, Union

import matplotlib.pyplot as plt
from pycocotools.coco import COCO

from datasetinsights.io.coco import load_coco_annotations
from datasetinsights.io.exceptions import InvalidCOCOImageIdError


def _load_img_ann_for_single_image(coco_obj: COCO, img_id: int):
    try:
        img = coco_obj.loadImgs(ids=[img_id])[0]
    except KeyError:
        raise InvalidCOCOImageIdError
    return img


def _load_image_from_img_ann(img_annotation: dict, data_dir: str):
    image_path = os.path.join(data_dir, img_annotation["file_name"])
    img = plt.imread(image_path)
    return img


def _load_annotations_for_single_img(coco_obj, img_id):
    ann_ids = coco_obj.getAnnIds(imgIds=img_id)
    annotations = coco_obj.loadAnns(ann_ids)
    return annotations


def display_single_img(coco_obj: COCO, img_id: int, data_dir: str):
    img_ann = _load_img_ann_for_single_image(coco_obj=coco_obj, img_id=img_id)
    img = _load_image_from_img_ann(img_annotation=img_ann, data_dir=data_dir)
    fig, ax = plt.subplots(dpi=100)
    ax.axis("off")
    ax.imshow(img)
    return fig


def display_ann_for_single_img(coco_obj: COCO, img_id: int, data_dir: str):
    img_ann = _load_img_ann_for_single_image(coco_obj=coco_obj, img_id=img_id)
    annotations = _load_annotations_for_single_img(
        coco_obj=coco_obj, img_id=img_ann["id"]
    )
    img = _load_image_from_img_ann(img_annotation=img_ann, data_dir=data_dir)
    fig, ax = plt.subplots(dpi=100)
    ax.axis("off")
    ax.imshow(img)
    coco_obj.showAnns(annotations, draw_bbox=True)
    return fig


def display_ann_for_all_img(
    annotation_file: str,
    data_dir: str,
    num_imgs: int = None,
    cat_id: Union[str, List] = 1,
):
    coco = load_coco_annotations(annotation_file=annotation_file)
    img_ids = coco.getImgIds(catIds=cat_id)
    if num_imgs:
        img_ids = random.sample(img_ids, k=num_imgs)
    for img_id in img_ids:
        fig, (ax1, ax2) = plt.subplots(1, 2, dpi=100, figsize=(18.5, 10.5))
        img_ann = _load_img_ann_for_single_image(coco_obj=coco, img_id=img_id)
        annotations = _load_annotations_for_single_img(
            coco_obj=coco, img_id=img_ann["id"]
        )
        img = _load_image_from_img_ann(
            img_annotation=img_ann, data_dir=data_dir
        )
        ax1.axis("off")
        ax1.imshow(img)
        ax2.axis("off")
        ax2.imshow(img)
        coco.showAnns(annotations, draw_bbox=True)
