import json
import math
from pathlib import Path

import pandas as pd
import pytest

from datasetinsights.stats.calculation import (
    convert_coco_annotations_to_df,
    get_bbox_heatmap,
    get_bbox_per_img_dict,
    get_bbox_relative_size_list,
    get_visible_keypoints_dict,
)
from datasetinsights.stats.visualization.constants import COCO_KEYPOINTS


@pytest.fixture()
def annotations_path():
    parent_dir = Path.cwd()
    json_file = (
        parent_dir
        / "tests"
        / "mock_data"
        / "coco"
        / "annotations"
        / "keypoints.json"
    )
    yield json_file
    json_file = None


@pytest.fixture()
def _setup_annotation_df(annotations_path):
    coco_json = json.load(open(annotations_path, "r"))

    df_image = pd.DataFrame(coco_json["images"])
    df_annotation = pd.DataFrame(coco_json["annotations"])

    df_coco = df_annotation.merge(df_image, left_on="image_id", right_on="id")
    yield df_coco
    df_coco = None


def test_convert_coco_annotations_to_df(annotations_path):
    processed_kp_dict = convert_coco_annotations_to_df(annotations_path)
    target_column_names = processed_kp_dict.columns.values.tolist()

    column_names = [
        "image_id",
        "area",
        "bbox",
        "iscrowd",
        "num_keypoints",
        "keypoints",
        "width",
        "height",
    ]

    for column_name in column_names:
        assert column_name in target_column_names


def test_get_bbox_heatmap(_setup_annotation_df):
    annotation_df = _setup_annotation_df
    bbox_heatmap = get_bbox_heatmap(annotation_df)
    height, width, _ = bbox_heatmap.shape

    max_width = max(annotation_df["width"])
    max_height = max(annotation_df["height"])

    assert max_width == width
    assert max_height == height
    assert (bbox_heatmap < 0).sum() == 0


def test_get_bbox_relative_size_list(_setup_annotation_df):
    annotation_df = _setup_annotation_df
    bbox_relative_size = get_bbox_relative_size_list(annotation_df)
    assert annotation_df.shape[0] == bbox_relative_size.shape[0]

    test_row = annotation_df.iloc[0]
    assert bbox_relative_size[0] == math.sqrt(
        test_row["area"] / (test_row["width"] * test_row["height"])
    )


def test_get_visible_keypoints_dict(_setup_annotation_df):
    keypoint_list = _setup_annotation_df["keypoints"].values.tolist()

    labeled_kpt_dict = get_visible_keypoints_dict(keypoint_list)
    for keypoint in COCO_KEYPOINTS:
        assert keypoint in labeled_kpt_dict.keys()
    for value in labeled_kpt_dict.values():
        assert value < 1 and value >= 0


def test_get_bbox_per_img_dict(_setup_annotation_df):
    annotation_df = _setup_annotation_df

    bbox_num_dict = get_bbox_per_img_dict(annotation_df)
    for value in bbox_num_dict.values():
        assert value < 1 and value >= 0
    assert sum(bbox_num_dict.values()) == 1
