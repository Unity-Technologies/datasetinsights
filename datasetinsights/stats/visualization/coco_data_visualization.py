import random
from typing import List, Union

import matplotlib.pyplot as plt
from pycocotools.coco import COCO

from datasetinsights.stats.coco_stats import (
    load_annotations_for_single_img,
    load_image_from_img_ann,
    load_img_ann_for_single_image,
)


def display_single_img(
    data_dir: str,
    img_id: int,
    annotation_file: str = None,
    coco_obj: COCO = None,
    show_annotation=False,
) -> plt.Figure:
    """
    Displays single image with the give image id
    Args:
        coco_obj (pycocotools.coco.COCO): COCO object
        annotation_file (str): COCO annotation json file.
        img_id (int): Image id of the image
        data_dir (str): Directory where data(images) is located
        show_annotation (bool): Show annotation for the image

    Returns:
        plt.Figure: Figure object

    """
    if coco_obj:
        coco = coco_obj
    elif annotation_file:
        coco = COCO(annotation_file=annotation_file)
    else:
        raise ValueError(
            "Must provide either annotation file or "
            "pycocotools.coco.COCO object"
        )
    img_ann = load_img_ann_for_single_image(coco_obj=coco, img_id=img_id)
    img = load_image_from_img_ann(img_annotation=img_ann, data_dir=data_dir)
    fig, ax = plt.subplots(dpi=100)
    ax.axis("off")
    ax.imshow(img)
    if show_annotation:
        annotations = load_annotations_for_single_img(
            coco_obj=coco, img_id=img_ann["id"]
        )
        coco.showAnns(annotations, draw_bbox=True)
    return fig


def display_ann_for_all_img(
    data_dir: str,
    annotation_file: str = None,
    coco_obj: COCO = None,
    num_imgs: int = None,
    cat_id: Union[int, List] = 1,
):
    """
    Plots annotations for all or specified number of images in the dataset

    Args:
        data_dir (str): Directory where data(images) is located
        annotation_file (str): COCO annotation json file.
        coco_obj (pycocotools.coco.COCO): COCO object
        num_imgs (int): Number of images to be displayed
        cat_id (Union[int, List]): List or int of category ids

    """
    if coco_obj:
        coco = coco_obj
    elif annotation_file:
        coco = COCO(annotation_file=annotation_file)
    else:
        raise ValueError(
            "Must provide either annotation file or "
            "pycocotools.coco.COCO object"
        )

    img_ids = coco.getImgIds(catIds=cat_id)
    if num_imgs:
        img_ids = random.sample(img_ids, k=num_imgs)
    for img_id in img_ids:
        fig, (ax1, ax2) = plt.subplots(1, 2, dpi=100, figsize=(18.5, 10.5))
        img_ann = load_img_ann_for_single_image(coco_obj=coco, img_id=img_id)
        annotations = load_annotations_for_single_img(
            coco_obj=coco, img_id=img_ann["id"]
        )
        img = load_image_from_img_ann(img_annotation=img_ann, data_dir=data_dir)
        ax1.axis("off")
        ax1.imshow(img)
        ax2.axis("off")
        ax2.imshow(img)
        coco.showAnns(annotations, draw_bbox=True)
