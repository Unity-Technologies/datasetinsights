import json
import pathlib

import numpy as np

from datasetinsights.stats.image_analysis import (
    get_bbox_fg_bg_var_laplacian,
    get_final_mask,
    get_seg_fg_bg_var_laplacian,
    laplacian_img,
)


def test_get_bbox_fg_bg_var_laplacian():
    cur_dir = pathlib.Path(__file__).parent.absolute()
    img_path = str(
        cur_dir
        / "mock_data"
        / "coco"
        / "images"
        / "camera_61855733451949387398181790757513827492.png"
    )
    ann_path = str(
        cur_dir / "mock_data" / "coco" / "annotations" / "keypoints.json"
    )
    laplacian = laplacian_img(img_path)
    f = open(ann_path)
    annotations = json.load(f)["annotations"]
    bbox_var_lap, img_var_lap = get_bbox_fg_bg_var_laplacian(
        laplacian, annotations
    )
    assert len(bbox_var_lap) > 0
    assert img_var_lap is not None


def test_get_seg_fg_bg_var_laplacian():
    laplacian = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    final_mask = np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]])
    expected_fg_var_lap = np.array([2, 5, 8]).var()
    expected_bg_var_lap = np.array([1, 3, 4, 6, 7, 9]).var()

    fg_var_lap, bg_var_lap = get_seg_fg_bg_var_laplacian(laplacian, final_mask)

    assert fg_var_lap == expected_fg_var_lap
    assert bg_var_lap == expected_bg_var_lap


def test_get_final_mask():
    mask_a = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    mask_b = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]])
    mask_c = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]])
    expected_final_mask = np.array([[1, 1, 1], [0, 0, 0], [0, 1, 0]])

    final_mask = get_final_mask(masks=[mask_a, mask_b, mask_c])

    assert np.array_equal(expected_final_mask, final_mask)
