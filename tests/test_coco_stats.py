from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from datasetinsights.constants import COCO_KEYPOINTS, COCO_SKELETON
from datasetinsights.io.coco import load_coco_annotations
from datasetinsights.io.exceptions import (
    InvalidCOCOCategoryIdError,
    InvalidCOCOImageIdError,
)
from datasetinsights.stats.coco_stats import (
    get_coco_keypoints,
    get_coco_skeleton,
    get_labeled_keypoints_dict,
    load_annotations_for_single_img,
    load_image_from_img_ann,
    load_img_ann_for_single_image,
)

parent_dir = Path(__file__).parent.absolute()
mock_coco_dir = parent_dir / "mock_data" / "coco"
coco_ann_file = mock_coco_dir / "annotations" / "keypoint_instances.json"
coco_img_dir = mock_coco_dir / "images"
coco = load_coco_annotations(annotation_file=str(coco_ann_file))


def test_coco_keypoints():
    kp = get_coco_keypoints(coco_obj=coco)
    assert kp == COCO_KEYPOINTS


def test_coco_keypoints_raise_error():
    with pytest.raises(InvalidCOCOCategoryIdError):
        get_coco_keypoints(coco_obj=coco, cat_id=66)


def test_coco_skeleton():
    kp = get_coco_skeleton(coco_obj=coco)
    assert kp == COCO_SKELETON


def test_load_img_annotation():
    expected_annotation = {
        "id": 1,
        "file_name": "camera_001.png",
        "width": 640,
        "height": 640,
    }

    assert expected_annotation == load_img_ann_for_single_image(
        coco_obj=coco, img_id=1
    )


def test_load_img_annotation_raises_error():
    with pytest.raises(InvalidCOCOImageIdError):
        load_img_ann_for_single_image(coco_obj=coco, img_id=123)


def test_load_img_from_img_ann():
    mock_img = Mock()
    mock_imread = MagicMock(return_value=mock_img)
    img_annotation = {
        "id": 1,
        "file_name": "camera_001.png",
        "width": 640,
        "height": 640,
    }

    with patch("datasetinsights.stats.coco_stats.plt.imread", mock_imread):
        expected_img_path = coco_img_dir / "camera_001.png"
        img = load_image_from_img_ann(
            img_annotation=img_annotation, data_dir=str(coco_img_dir)
        )
        assert img == mock_img
        assert mock_imread.call_args_list[0][0][0] == str(expected_img_path)


def test_load_ann_for_single_img():
    mock_annotations = Mock()
    mock_load_ann = MagicMock(return_value=mock_annotations)

    with patch("datasetinsights.stats.coco_stats.COCO.loadAnns", mock_load_ann):
        ann = load_annotations_for_single_img(coco_obj=coco, img_id=1)
        assert ann == mock_annotations


def test_get_labeled_kp_dict():
    expected_kp_dict = {
        "nose": 0.625,
        "left_eye": 0.625,
        "right_eye": 0.625,
        "left_ear": 0.625,
        "right_ear": 0.625,
        "left_shoulder": 0.625,
        "right_shoulder": 0.625,
        "left_elbow": 0.625,
        "right_elbow": 0.625,
        "left_wrist": 0.625,
        "right_wrist": 0.75,
        "left_hip": 0.625,
        "right_hip": 0.625,
        "left_knee": 0.625,
        "right_knee": 0.625,
        "left_ankle": 0.875,
        "right_ankle": 0.875,
    }
    assert expected_kp_dict == get_labeled_keypoints_dict(
        annotation_file=str(coco_ann_file)
    )
