import math
import random

import numpy as np
import pandas as pd
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


def _get_bbox_heatmap(coco_img_ids, coco_obj):
    # calculate bbox shape according to largest image in the dataset
    max_height, max_width = 0, 0
    for idx in coco_img_ids:
        img = coco_obj.loadImgs(idx)[0]
        max_height = max(img["height"], max_height)
        max_width = max(img["width"], max_width)

    bbox_heatmap = np.zeros([max_height, max_width, 1])
    return bbox_heatmap


def _get_kpt_dict(coco_obj):
    kpt_dict = {}
    coco_kpts = coco_obj.loadCats(coco_obj.getCatIds())[0]["keypoints"]
    for key in coco_kpts:
        kpt_dict[key] = 0
    return kpt_dict


def _get_bbox_dict():
    bbox_dict = {}
    for key in COCO_AREA_DICT:
        bbox_dict[key] = 0
    return bbox_dict


def get_stats(img_ids, cat_ids, coco_obj):
    total_instances = 0
    bbox_relative_size = []

    kpt_dict = _get_kpt_dict(coco_obj)
    bbox_dict = _get_bbox_dict()
    bbox_heatmap = _get_bbox_heatmap(coco_img_ids=img_ids, coco_obj=coco_obj)

    keypoints = list(kpt_dict.keys())

    for idx in tqdm(img_ids):
        img = coco_obj.loadImgs(idx)[0]
        h, w = img["height"], img["width"]
        annotation_ids = coco_obj.getAnnIds(
            imgIds=img["id"], catIds=cat_ids, iscrowd=None
        )
        annotations = coco_obj.loadAnns(annotation_ids)

        for person in range(len(annotations)):
            total_instances += 1
            bbox = np.array(annotations[person]["bbox"]).astype(int)
            bbox_heatmap[
                bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2], :
            ] += 1

            kp_visbility = annotations[person]["keypoints"][2::3]
            for i in range(len(keypoints)):
                if kp_visbility[i] != 0:
                    kpt_dict[keypoints[i]] += 1

            # statistics of bbox sizes COCO format
            area_size = annotations[person]["area"]

            for ind, ar in enumerate(COCO_AREA_RANGES):
                if (area_size >= ar[0]) and (area_size < ar[1]):
                    bbox_dict[ind] += 1

            # statistics of bbox sizes
            area_size = annotations[person]["area"]
            area_size = area_size / (h * w)
            relative_size = math.sqrt(area_size)
            bbox_relative_size.append(relative_size)

    for key in kpt_dict.keys():
        kpt_dict[key] = kpt_dict[key] / total_instances

    return kpt_dict, bbox_dict, bbox_heatmap, bbox_relative_size


def convert_coco_obj_to_df(coco_obj, cat_ids, max_frames=None):
    images_data = []
    persons_data = []

    ids = coco_obj.getImgIds(catIds=cat_ids)

    if max_frames:
        ids = random.sample(ids, max_frames)

    for i, img_id in enumerate(ids):
        img_meta = coco_obj.imgs[img_id]
        ann_ids = coco_obj.getAnnIds(imgIds=img_id)

        # basic parameters of an image
        img_file_name = img_meta["file_name"]
        w = img_meta["width"]
        h = img_meta["height"]
        # retrieve metadata for all persons in the current image
        meta = coco_obj.loadAnns(ann_ids)

        images_data.append(
            {
                "image_id": int(img_id),
                "path": img_file_name,
                "width": int(w),
                "height": int(h),
            }
        )
        # iterate over all metadata
        for m in meta:
            persons_data.append(
                {
                    "image_id": m["image_id"],
                    "is_crowd": m["iscrowd"],
                    "bbox": m["bbox"],
                    "area": m["area"],
                    "num_keypoints": m["num_keypoints"],
                    "keypoints": m["keypoints"],
                }
            )
    # create dataframe with image paths
    images_df = pd.DataFrame(images_data)
    images_df.set_index("image_id", inplace=True)

    # create dataframe with persons
    persons_df = pd.DataFrame(persons_data)
    persons_df.set_index("image_id", inplace=True)

    return images_df, persons_df
