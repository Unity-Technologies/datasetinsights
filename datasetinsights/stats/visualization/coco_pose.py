import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import collections as mc

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


def _load_annotations_from_json(json_file):
    f = open(json_file)
    data = json.load(f)
    annotations = data["annotations"]
    return annotations


def _is_torso_visible_or_labelled(kp):
    return (
        (kp[17] == 1 or kp[17] == 2)
        and (kp[20] == 1 or kp[20] == 2)
        and (kp[41] == 1 or kp[41] == 2)
        and (kp[38] == 1 or kp[38] == 2)
    )


def _get_ann_where_torso_visible(annotations):
    visible_kp_annotations = []
    for ann in annotations:
        kp = ann["keypoints"]
        if _is_torso_visible_or_labelled(kp):
            visible_kp_annotations.append(ann)
    return visible_kp_annotations


def _calc_mid(p1, p2):
    return (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2


def _calc_dist(p1, p2):
    return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))


def _remove_visibility_flag_from_kp(kp):
    updated_kp = list(kp)
    del updated_kp[2::3]
    return updated_kp


def _translate_kp(x, y, tx, ty):
    x = [number - tx if number > 0.0 else 0.0 for number in x]
    y = [number - ty if number > 0.0 else 0.0 for number in y]
    return x, y


def _scale_kp(x, y, sf):
    x = [number / sf for number in x]
    y = [number / sf for number in y]
    return x, y


def _translate_and_scale_xy_kp(x, y):

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


def process_annotations(json_file):

    annotations = _load_annotations_from_json(json_file)
    annotations = _get_ann_where_torso_visible(annotations)

    kp_dict = {}
    for name in COCO_KEYPOINTS:
        kp_dict[name] = {"x": [], "y": []}

    for ann in annotations:
        keypoints = ann["keypoints"]
        keypoints = _remove_visibility_flag_from_kp(keypoints)

        # Separate x and y keypoints
        x_kp, y_kp = keypoints[0::2], keypoints[1::2]
        x_kp, y_kp = _translate_and_scale_xy_kp(x_kp, y_kp)

        # save keypoints to dict
        idx = 0
        for xi, yi in zip(x_kp, y_kp):
            if xi == 0 and yi == 0:
                pass
            else:
                kp_dict[COCO_KEYPOINTS[idx]]["x"].append(xi)
                kp_dict[COCO_KEYPOINTS[idx]]["y"].append(yi)
            idx += 1

    return kp_dict


def generate_scatter_plot(kp_dict, title="", fig_path=None):
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
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")

    # Turn off tick labels
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_title(title)

    # Invert axes
    ax.invert_yaxis()
    ax.set_aspect("equal")

    plt.legend(
        loc="lower center", bbox_to_anchor=(0.5, -0.17), ncol=5, fancybox=True,
    )

    if fig_path:
        plt.savefig(fig_path, bbox_inches="tight", pad_inches=0.15)
    plt.show()


def generate_heatmaps(kp_dict, color="red", title="", fig_dest=None):

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
        ax.spines["right"].set_color("none")
        ax.spines["top"].set_color("none")

        # Turn off tick labels
        ax.set_yticklabels([])
        ax.set_xticklabels([])

        props = dict(
            boxstyle="square", facecolor="wheat", alpha=0.1, edgecolor="black"
        )
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

        if fig_dest:
            plt.savefig(
                os.path.join(fig_dest, f"{name}_heatmap.png"),
                bbox_inches="tight",
                pad_inches=0.15,
            )

        plt.show()


def _get_avg_kp(kp_dict):
    x_avg, y_avg = [], []
    for name in COCO_KEYPOINTS:
        kp_x = np.array(kp_dict[name]["x"])
        kp_y = np.array(kp_dict[name]["y"])
        x_avg.append(np.mean(kp_x))
        y_avg.append(np.mean(kp_y))
    return x_avg, y_avg


def _get_skeleton(x, y):
    s = []
    for p1, p2 in COCO_SKELETON:
        p1 -= 1
        p2 -= 1
        s.append([(x[p1], y[p1]), (x[p2], y[p2])])
    return s


def generate_avg_skeleton(kp_dict, title="", fig_path=None, scatter=False):
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
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")

    # Turn off tick labels
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    ax.set_title(title)

    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()

    if scatter:
        plt.scatter(x_avg, y_avg, s=7)

    fig.tight_layout(pad=0.2)
    if fig_path:
        plt.savefig(fig_path, bbox_inches="tight", pad_inches=0.15)
    plt.show()
