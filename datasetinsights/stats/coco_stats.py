import math

import numpy as np
import pandas as pd
import pycocotools.coco
from pycocotools.coco import COCO
from tqdm import tqdm

COCO_AREA_RANGES = [
    [0 ** 2, 1e5 ** 2],  # all
    [0 ** 2, 32 ** 2],  # small
    [32 ** 2, 96 ** 2],  # medium
    [96 ** 2, 1e5 ** 2],  # large
    [96 ** 2, 128 ** 2],  # 96-128
    [128 ** 2, 256 ** 2],  # 128-256
    [256 ** 2, 512 ** 2],  # 256-512
    [512 ** 2, 1e5 ** 2],
]  # 512-inf

COCO_AREA_DICT = {
    0: "all",
    1: "small",
    2: "medium",
    3: "large",
    4: "96-128",
    5: "128-256",
    6: "256-512",
    7: "512-inf",
}


def _get_empty_bbox_heatmap(coco_obj):
    # calculate bbox shape according to largest image in the dataset
    coco_img_ids = coco_obj.getImgIds()
    max_height, max_width = 0, 0
    for idx in coco_img_ids:
        img = coco_obj.loadImgs(idx)[0]
        max_height = max(img["height"], max_height)
        max_width = max(img["width"], max_width)

    bbox_heatmap = np.zeros([max_height, max_width, 1])
    return bbox_heatmap


def _get_labeled_kpt_dict(coco_obj: pycocotools.coco.COCO):
    kpt_dict = {}
    coco_kpts = coco_obj.loadCats(coco_obj.getCatIds())[0]["keypoints"]
    for key in coco_kpts:
        kpt_dict[key] = 0.0
    return kpt_dict


def _get_bbox_area_dict():
    bbox_dict = {}
    for key in COCO_AREA_DICT:
        bbox_dict[key] = 0
    return bbox_dict


def _get_bbox_relative_size(bbox_area, image_h, image_w):
    bbox_relative_size = math.sqrt(bbox_area / (image_h * image_w))
    return bbox_relative_size


def get_labeled_keypoints(annotation_file):
    coco = COCO(annotation_file)
    labeled_kpt_dict = _get_labeled_kpt_dict(coco_obj=coco)
    keypoints = list(labeled_kpt_dict.keys())
    img_ids = coco.getImgIds()
    cat_ids = coco.getCatIds()
    total_instances = 0

    for idx in tqdm(img_ids):
        img = coco.loadImgs(idx)[0]
        annotation_ids = coco.getAnnIds(
            imgIds=img["id"], catIds=cat_ids, iscrowd=None
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
    coco = COCO(annotation_file)
    img_ids = coco.getImgIds()
    cat_ids = coco.getCatIds()
    bbox_heatmap = _get_empty_bbox_heatmap(coco_obj=coco)

    for idx in tqdm(img_ids):
        img = coco.loadImgs(idx)[0]
        annotation_ids = coco.getAnnIds(
            imgIds=img["id"], catIds=cat_ids, iscrowd=None
        )
        annotations = coco.loadAnns(annotation_ids)

        for person in range(len(annotations)):
            bbox = np.array(annotations[person]["bbox"]).astype(int)
            bbox_heatmap[
                bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2], :
            ] += 1

    return bbox_heatmap


def bbox_relative_size_list(annotation_file):
    coco = COCO(annotation_file)
    img_ids = coco.getImgIds()
    cat_ids = coco.getCatIds()
    bbox_relative_size = []

    for idx in tqdm(img_ids):
        img = coco.loadImgs(idx)[0]
        h, w = img["height"], img["width"]
        annotation_ids = coco.getAnnIds(
            imgIds=img["id"], catIds=cat_ids, iscrowd=None
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
    img_ids = coco.getImgIds()

    coco_data = []

    for i, img_id in enumerate(img_ids):
        img_meta = coco.imgs[img_id]
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
