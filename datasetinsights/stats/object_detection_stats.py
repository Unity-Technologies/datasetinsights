import json
from typing import Dict

import numpy as np
import pandas as pd


def convert_coco_annotations_to_df(filename: str) -> pd.DataFrame:
    """
    Converts coco annotation file to pandas df for processing.
    Args:
        filename (str): Annotation file path
    Returns:
        coco dataframe (pd.DataFrame): dataframe of annotation info.
    """
    coco_json = json.load(open(filename, "r"))

    df_image = pd.DataFrame(coco_json["images"])
    df_annotation = pd.DataFrame(coco_json["annotations"])

    df_coco = df_annotation.merge(df_image, left_on="image_id", right_on="id")

    return df_coco


def get_bbox_relative_size_list(annotation_df: pd.DataFrame) -> np.ndarray:
    """
    Args:
        annotation_df (pd.DataFrame): dataframe with image and
        bbox_annotation in each row,(columns include: width
        (image width), height (image height), area (bbox size))
    Returns:
        bbox_relative_size_list (np.ndarray): List of all bbox
         sizes relative to its image size
    """
    bbox_size = annotation_df["area"]
    image_size = annotation_df["width"] * annotation_df["height"]
    bbox_relative_size = np.sqrt(bbox_size / image_size)

    return bbox_relative_size


def get_bbox_heatmap(annotation_df: pd.DataFrame) -> np.ndarray:
    """
    Args:
        annotation_df (pd.DataFrame): dataframe with image
        and bbox_annotation in each row, (columns include:
        width (image width), height (image height),
        bbox ([top_left_x, top_left_y, width, height]))
    Returns:
        bbox_heatmap (np.ndarray): numpy array of size of
        the max sized image in the dataset with values describing
        bbox intensity over the entire dataset images
        at a particular pixel.
    """
    max_width = max(annotation_df["width"])
    max_height = max(annotation_df["height"])
    bbox_heatmap = np.zeros([max_height, max_width, 1])

    for bbox in annotation_df["bbox"]:
        bbox = np.array(bbox).astype(int)
        bbox_heatmap[
            bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2], :
        ] += 1

    return bbox_heatmap


def get_bbox_per_img_dict(annotation_df: pd.DataFrame) -> Dict:
    """
    Args:
        annotation_df (pd.DataFrame): dataframe with each annotation
        in each row, (columns include: iscrowd (bool), image_id (image id))
    Returns:
        Dict: Dictionary of number of bbox per image where key is the number
        of bbox and val is the probability of that number of bbox images in
        the dataset.
    """
    annotated_persons_df = annotation_df[(annotation_df["iscrowd"] == 0)]

    persons_in_img_df = pd.DataFrame(
        {"cnt": annotated_persons_df[["image_id"]].value_counts()}
    )
    persons_in_img_df.reset_index(level=[0], inplace=True)

    # group by counter so we will get the dataframe with number of
    # annotated people in a single image

    persons_in_img_cnt_df = persons_in_img_df.groupby(["cnt"]).count()

    # extract arrays
    x_occurences = persons_in_img_cnt_df.index.values
    y_images = persons_in_img_cnt_df["image_id"].values
    total_images = sum(y_images)

    bbox_num_dict = {}
    for key, value in zip(x_occurences, y_images):
        bbox_num_dict[key] = value / total_images
    return bbox_num_dict
