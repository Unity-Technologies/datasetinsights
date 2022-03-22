import json
from pathlib import Path

import pytest

from datasetinsights.stats.image_analysis import (
    get_average_skeleton,
    get_scale_keypoints,
)
from datasetinsights.stats.visualization.constants import (
    COCO_KEYPOINTS,
    COCO_SKELETON,
)


@pytest.fixture()
def _setup_annotations():
    parent_dir = Path.cwd()
    json_file = (
        parent_dir
        / "tests"
        / "mock_data"
        / "coco"
        / "annotations"
        / "keypoints.json"
    )
    f = open(json_file)
    data = json.load(f)
    annotations = data["annotations"]
    keypoints_list = []
    for k in annotations:
        keypoints_list.append(k["keypoints"])
    yield keypoints_list
    keypoints_list = None


def test_get_scale_keypoints(_setup_annotations):
    annotations = _setup_annotations
    processed_kp_dict = get_scale_keypoints(annotations)

    assert set(COCO_KEYPOINTS).issubset(set(processed_kp_dict.keys()))
    for keypoint in COCO_KEYPOINTS:
        count = sum(
            map(lambda x: x > 2.5 or x < -2.5, processed_kp_dict[keypoint]["x"])
        )
        assert count == 0
        count = sum(
            map(lambda x: x > 2.5 or x < -2.5, processed_kp_dict[keypoint]["y"])
        )
        assert count == 0


def test_get_scale_keypoints_bad_case():
    annotations = [[0] * 40, [1] * 60]
    with pytest.raises(ValueError):
        get_scale_keypoints(annotations)


@pytest.fixture()
def _setup_kp_dict():
    kp_dict = {}
    for name in COCO_KEYPOINTS:
        kp_dict[name] = {"x": [2, 0], "y": [0, 2]}
    yield kp_dict
    kp_dict = None


def test_get_average_skeleton(_setup_kp_dict):
    kp_dict = _setup_kp_dict
    kp_link_list = get_average_skeleton(kp_dict)

    assert kp_link_list[0] == [(1, 1), (1, 1)]
    assert len(kp_link_list) == len(COCO_SKELETON)
