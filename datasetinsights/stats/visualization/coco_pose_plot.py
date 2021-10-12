import math
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import collections as mc

from datasetinsights.io.coco import load_coco_annotations_json

COCO_SKELETON = [
    [16, 14],
    [14, 12],
    [17, 15],
    [15, 13],
    [12, 13],
    [6, 12],
    [7, 13],
    [6, 7],
    [6, 8],
    [7, 9],
    [8, 10],
    [9, 11],
    [2, 3],
    [1, 2],
    [1, 3],
    [2, 4],
    [3, 5],
    [4, 6],
    [5, 7],
]

COCO_KEYPOINTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def _is_torso_visible_or_labeled(kp: List) -> bool:
    """

    Args:
        kp (list): List of keypoints

    Returns: True if torso (left hip, right hip, left shoulder,
    right shoulder) is visible else False

    """
    return (
        (kp[17] == 1 or kp[17] == 2)
        and (kp[20] == 1 or kp[20] == 2)
        and (kp[41] == 1 or kp[41] == 2)
        and (kp[38] == 1 or kp[38] == 2)
    )


def _get_kp_where_torso_visible(annotations: Dict) -> List:
    """Return list of keypoint where torso is visible or labeled"""
    keypoints = []
    for ann in annotations:
        kp = ann["keypoints"]
        if _is_torso_visible_or_labeled(kp):
            keypoints.append(kp)
    return keypoints


def _calc_mid(p1: Tuple[Any, Any], p2: Tuple[Any, Any]):
    """
    Calculate mid point of two points
    Args:
        p1 (Tuple[float]): Point 1
        p2 (Tuple[float]): Point 2

    Returns: x,y value of mid point
    """
    return (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2


def _calc_dist(p1: Tuple[Any, Any], p2: Tuple[Any, Any]) -> float:
    """
    Args:
        p1 (Tuple[float]): Point 1
        p2 (Tuple[float]): Point 2

    Returns:
        float: Distance between two points

    """
    return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))


def _remove_visibility_flag_from_kp(kp: List) -> List:
    """Removes visible flag which is every third element in the keypoints
    list."""
    updated_kp = list(kp)
    del updated_kp[2::3]
    return updated_kp


def _translate_kp(x: List, y: List, tx: float, ty: float) -> Tuple[List, List]:
    """

    Args:
        x (List): List of x keypoints
        y (List): List of y keypoints
        tx (float): Value of translation in x direction
        ty (float): Value of translation in y direction

    Returns:
        Tuple[List, List]: List of x and y translated keypoints

    """
    x = np.array(x)
    y = np.array(y)
    x = x - tx
    y = y - ty
    return x.tolist(), y.tolist()


def _scale_kp(x: List, y: List, sf: float) -> Tuple[List, List]:
    """

    Args:
        x (List): List of x keypoints
        y (List): List of y keypoints
        sf (float): Scale value

    Returns:
        Tuple[List, List]: List of x and y scaled keypoints
    """
    x = np.array(x)
    y = np.array(y)
    x = x / sf
    y = y / sf
    return x.tolist(), y.tolist()


def _translate_and_scale_xy_kp(x: List, y: List) -> Tuple[List, List]:
    """Return x and y keypoints after performing translation and scaling."""

    left_hip, right_hip = (x[11], y[11]), (x[12], y[12])
    left_shoulder, right_shoulder = (x[5], y[5]), (x[6], y[6])

    # Translate all points according to mid_hip being at 0,0
    mid_hip = _calc_mid(right_hip, left_hip)
    x, y = _translate_kp(x, y, mid_hip[0], mid_hip[1])

    # Calculate scale factor
    scale = (
        _calc_dist(left_shoulder, left_hip)
        + _calc_dist(right_shoulder, right_hip)
    ) / 2
    x, y = _scale_kp(x, y, scale)

    return x, y


def _process_annotations(annotation_file: str) -> Dict:
    """
    Process annotation file to extract information for pose plots.
    Args:
        annotation_file (JSON): COCO format json annotation file path

    Returns:
        Dict: Processed keypoint dict useful for plotting pose plots.

    """

    annotations = load_coco_annotations_json(annotation_file=annotation_file)
    keypoints = _get_kp_where_torso_visible(annotations)

    processed_kp_dict = {}
    for name in COCO_KEYPOINTS:
        processed_kp_dict[name] = {"x": [], "y": []}

    for kp in keypoints:
        kp = _remove_visibility_flag_from_kp(kp)

        # Separate x and y keypoints
        x_kp, y_kp = kp[0::2], kp[1::2]
        x_kp, y_kp = _translate_and_scale_xy_kp(x_kp, y_kp)

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


def _eliminate_axes(axes: List[str], ax: plt.Axes):
    for axis in axes:
        ax.spines[axis].set_color("none")


def _turn_off_x_tick_labels(ax: plt.Axes):
    ax.set_xticklabels([])


def _turn_off_y_tick_labels(ax: plt.Axes):
    ax.set_yticklabels([])


def _turn_off_xy_tick_labels(ax: plt.Axes):
    _turn_off_x_tick_labels(ax=ax)
    _turn_off_y_tick_labels(ax=ax)


def save_figure(fig: plt.Figure, fig_path: str):
    """
    Args:
        fig (plt.Figure): Figure object
        fig_path (str): Path where figure is to be saved
    """
    fig.savefig(fname=fig_path, bbox_inches="tight", pad_inches=0.15)


def generate_scatter_plot(annotation_file: str, title: str = "",) -> plt.Figure:
    """

    Args:
        annotation_file (JSON): COCO format json annotation file path
        title (str): Title of the plot

    Returns:
        plt.Figure: Figure object

    """
    kp_dict = _process_annotations(annotation_file=annotation_file)
    fig, ax = plt.subplots(dpi=300, figsize=(8, 8))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(COCO_KEYPOINTS)))[::-1]
    i = 0
    for name in COCO_KEYPOINTS:
        plt.scatter(
            kp_dict[name]["x"],
            kp_dict[name]["y"],
            color=colors[i],
            label=name,
            s=3,
        )
        i += 1

    plt.xlim(-2, 2)
    plt.ylim(-2, 2)

    ax.spines["left"].set_position("center")
    ax.spines["bottom"].set_position("center")

    # Eliminate upper and right axes
    _eliminate_axes(axes=["upper", "right"], ax=ax)

    # Turn off tick labels
    _turn_off_xy_tick_labels(ax=ax)
    ax.set_title(title)

    # Invert axes
    ax.invert_yaxis()
    ax.set_aspect("equal")

    plt.legend(
        loc="lower center", bbox_to_anchor=(0.5, -0.17), ncol=5, fancybox=True,
    )

    return fig


def generate_heatmaps(
    annotation_file: str, color: str = "red",
) -> List[plt.Figure]:
    """

    Args:
        annotation_file (JSON): COCO format json annotation file path
        color (str): Color of the heatmap

    Returns:
        plt.Figure: Figure object
    """
    kp_dict = _process_annotations(annotation_file=annotation_file)

    figures = []

    for name in COCO_KEYPOINTS:
        fig, ax = plt.subplots(dpi=100, figsize=(8, 8))
        sns.kdeplot(
            x=kp_dict[name]["x"],
            y=kp_dict[name]["y"],
            shade=True,
            label=name,
            ax=ax,
            alpha=0.85,
            color=color,
            cbar=False,
        )

        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)

        ax.spines["left"].set_position("center")
        ax.spines["bottom"].set_position("center")

        # Eliminate upper and right axes
        _eliminate_axes(axes=["upper", "right"], ax=ax)

        # Turn off tick labels
        _turn_off_xy_tick_labels(ax=ax)

        t = name.split("_")
        t = [x.capitalize() for x in t]
        t = " ".join(t)
        textstr = t
        ax.text(
            0.005, 0.080, textstr, transform=ax.transAxes, fontsize=36, va="top"
        )

        ax.patch.set_edgecolor("black")

        ax.patch.set_linewidth("1")

        # Invert axes
        ax.invert_yaxis()

        ax.set_aspect("equal")

        figures.append(fig)

    return figures


def _get_avg_kp(kp_dict):
    x_avg, y_avg = [], []
    for name in COCO_KEYPOINTS:
        kp_x = np.array(kp_dict[name]["x"])
        kp_y = np.array(kp_dict[name]["y"])
        x_avg.append(np.mean(kp_x))
        y_avg.append(np.mean(kp_y))
    return x_avg, y_avg


def _get_skeleton(x_kp, y_kp):
    s = []
    for p1, p2 in COCO_SKELETON:
        s.append([(x_kp[p1 - 1], y_kp[p1 - 1]), (x_kp[p2 - 1], y_kp[p2 - 1])])
    return s


def generate_avg_skeleton(
    annotation_file: str, title: str = "", scatter: bool = False
) -> plt.Figure:
    """

    Args:
        annotation_file (JSON): COCO format json annotation file path
        title (str): Title of the plot
        scatter (bool): Overlay scatter plot on average skeleton

    Returns:
        plt.Figure: Figure object
    """
    kp_dict = _process_annotations(annotation_file=annotation_file)

    x_avg, y_avg = _get_avg_kp(kp_dict)
    skeleton = _get_skeleton(x_avg, y_avg)

    c = plt.cm.rainbow(np.linspace(0, 1, len(skeleton)))
    lc = mc.LineCollection(skeleton, colors=c, linewidths=1)

    fig, ax = plt.subplots(dpi=300, figsize=(4, 4))
    ax.add_collection(lc)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

    ax.spines["left"].set_position("center")
    ax.spines["bottom"].set_position("center")

    # Eliminate upper and right axes
    _eliminate_axes(axes=["upper", "right"], ax=ax)

    # Turn off tick labels
    _turn_off_xy_tick_labels(ax=ax)

    ax.set_title(title)

    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()

    if scatter:
        plt.scatter(x_avg, y_avg, s=7)

    fig.tight_layout(pad=0.2)

    return fig
