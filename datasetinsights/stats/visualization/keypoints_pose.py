import math
from typing import Any, Dict, List, Tuple

import numpy as np

from datasetinsights.stats.visualization.constants import (
    COCO_KEYPOINTS,
    COCO_SKELETON,
)


def _is_torso_visible_or_labeled(kp: List) -> bool:
    """
    True if torso (left hip, right hip, left shoulder,
    right shoulder) is visible else False
    """
    return (
        (kp[17] == 1 or kp[17] == 2)
        and (kp[20] == 1 or kp[20] == 2)
        and (kp[41] == 1 or kp[41] == 2)
        and (kp[38] == 1 or kp[38] == 2)
    )


def _get_kp_where_torso_visible(annotations: List) -> List:
    """
    List of keypoint where torso is visible or labeled
    """
    keypoints = []
    for ann in annotations:
        if _is_torso_visible_or_labeled(ann):
            keypoints.append(ann)
    return keypoints


def _calc_mid(p1: Tuple[Any, Any], p2: Tuple[Any, Any]):
    """
    Calculate mid point of two points
    """
    return (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2


def _calc_dist(p1: Tuple[Any, Any], p2: Tuple[Any, Any]) -> float:
    """
    Calculate distance between two points
    """
    return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))


def _translate_and_scale_xy(X: np.ndarray, Y: np.ndarray):
    """
    Return keypoints axis list X and Y after performing translation and scaling.
    """
    left_hip, right_hip = (X[11], Y[11]), (X[12], Y[12])
    left_shoulder, right_shoulder = (X[5], Y[5]), (X[6], Y[6])

    # Translate all points according to mid_hip being at 0,0
    mid_hip = _calc_mid(right_hip, left_hip)
    X = np.where(X > 0.0, X - mid_hip[0], 0.0)
    Y = np.where(Y > 0.0, Y - mid_hip[1], 0.0)

    # Calculate scale factor
    scale = (
        _calc_dist(left_shoulder, left_hip)
        + _calc_dist(right_shoulder, right_hip)
    ) / 2

    return X / scale, Y / scale


def get_scale_keypoints(annotations: List) -> Dict:
    """
    Process keypoints annotations to extract information for pose plots.
    Args:
        annotations (list): List of keypoints lists with format
        [x1, y1, v1, x2, y2, v2, ...] with the order of COCO_KEYPOINTS
    Returns:
        Dict: Processed key-value pair of keypoints name -> (x,y) list.
    """
    keypoints = _get_kp_where_torso_visible(annotations)

    processed_kp_dict = {}
    for name in COCO_KEYPOINTS:
        processed_kp_dict[name] = {"x": [], "y": []}

    for kp in keypoints:
        # Separate x and y keypoints
        x_kp, y_kp = np.array(kp[0::3]), np.array(kp[1::3])
        x_kp, y_kp = _translate_and_scale_xy(x_kp, y_kp)

        # save keypoints to dict
        idx = 0
        for xi, yi in zip(x_kp, y_kp):
            if xi == 0 and yi == 0:
                pass
            elif xi > 2.5 or xi < -2.5 or yi > 2.5 or yi < -2.5:
                pass
            else:
                processed_kp_dict[COCO_KEYPOINTS[idx]]["x"].append(xi)
                processed_kp_dict[COCO_KEYPOINTS[idx]]["y"].append(yi)
            idx += 1

    return processed_kp_dict


def _get_avg_kp(kp_dict: Dict):
    """
    Return average value of keypoints axis list X and Y.
    """
    x_avg, y_avg = [], []
    for key in COCO_KEYPOINTS:
        kp_x = np.array(kp_dict[key]["x"])
        kp_y = np.array(kp_dict[key]["y"])
        x_avg.append(np.mean(kp_x))
        y_avg.append(np.mean(kp_y))
    return x_avg, y_avg


def get_average_skeleton(kp_dict: Dict, skeleton=COCO_SKELETON) -> List:
    """
    return skeleton (a list of connected human joints) of
    average keypoints values.
    Args:
        kp_dict (dict): key-value pair of keypoints name -> (x,y) list
    Returns:
        list: list of skeleton connections.
    """
    x, y = _get_avg_kp(kp_dict)
    s = []
    for p1, p2 in skeleton:
        s.append([(x[p1 - 1], y[p1 - 1]), (x[p2 - 1], y[p2 - 1])])
    return s
