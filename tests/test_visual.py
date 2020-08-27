import pathlib
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from pytest import approx

from datasetinsights.datasets.cityscapes import CITYSCAPES_COLOR_MAPPING
from datasetinsights.io.bbox import BBox2D
from datasetinsights.stats.visualization.bbox2d_plot import (
    _COLOR_NAME_TO_RGB,
    _add_single_bbox_on_image,
    add_single_bbox_on_image,
)
from datasetinsights.stats.visualization.plots import (
    _convert_euler_rotations_to_scatter_points,
    bar_plot,
    decode_segmap,
    histogram_plot,
    match_boxes,
    plot_bboxes,
)


def test_decode_segmap():
    ids = list(CITYSCAPES_COLOR_MAPPING.keys())
    colors = list(CITYSCAPES_COLOR_MAPPING.values())
    img = np.array([ids] * 2)
    color_img = np.array([colors] * 2) / 255.0

    assert decode_segmap(img) == approx(color_img)


def test_histogram_plot():
    df = pd.DataFrame({"x": [1, 2, 3]})
    mock_figure = Mock()
    mock_layout = Mock()
    mock_figure.update_layout = MagicMock(return_value=mock_layout)
    mock_histogram_plot = MagicMock(return_value=mock_figure)

    with patch(
        "datasetinsights.stats.visualization.plots.px.histogram",
        mock_histogram_plot,
    ):
        fig = histogram_plot(df, x="x")
        assert fig == mock_layout


def test_bar_plot():
    df = pd.DataFrame({"x": ["a", "b", "c"], "y": [1, 2, 3]})
    mock_figure = Mock()
    mock_layout = Mock()
    mock_figure.update_layout = MagicMock(return_value=mock_layout)
    mock_bar_plot = MagicMock(return_value=mock_figure)

    with patch(
        "datasetinsights.stats.visualization.plots.px.bar", mock_bar_plot
    ):
        fig = bar_plot(df, x="x", y="y")
        assert fig == mock_layout


def test_convert_euler_rotations_to_scatter_points():
    df = pd.DataFrame({"x": [0, 90, 0], "y": [0, 0, 90]})
    expected = [
        [0, 1, 0, "x: 0°  y: 0°"],
        [-1, 0, 0, "x: 90°  y: 0°"],
        [0, 0, 1, "x: 0°  y: 90°"],
    ]
    points = list(_convert_euler_rotations_to_scatter_points(df, "x", "y"))
    assert all(
        [
            approx(expected[0], actual[0])
            and approx(expected[1], actual[1])
            and approx(expected[2], actual[2])
            and expected[3] == actual[3]
            for expected, actual in zip(expected, points)
        ]
    )


def test_convert_euler_rotations_to_scatter_points_with_z():
    df = pd.DataFrame(
        {"x": [0, 90, 0, 0], "y": [0, 0, 90, 0], "z": [0, 0, 0, 90]}
    )
    expected = [
        [0, 1, 0, "x: 0°  y: 0°  z: 0°"],
        [-1, 0, 0, "x: 90°  y: 0°  z: 0°"],
        [0, 0, 1, "x: 0°  y: 90°  z: 0°"],
        [0, 1, 0, "x: 0°  y: 0°  z: 90°"],
    ]
    points = list(_convert_euler_rotations_to_scatter_points(df, "x", "y", "z"))
    assert all(
        [
            approx(expected[0], actual[0])
            and approx(expected[1], actual[1])
            and approx(expected[2], actual[2])
            and expected[3] == actual[3]
            for expected, actual in zip(expected, points)
        ]
    )


def test_plot_bboxes():
    cur_dir = pathlib.Path(__file__).parent.absolute()
    img = Image.open(
        str(cur_dir / "mock_data" / "simrun" / "captures" / "camera_000.png")
    )
    label_mappings = {1: "car", 2: "tree", 3: "light"}
    boxes = [
        BBox2D(label=1, x=1, y=1, w=2, h=3),
        BBox2D(label=1, x=7, y=6, w=3, h=4),
        BBox2D(label=1, x=2, y=6, w=2, h=4),
    ]
    colors = ["green", "red", "green"]

    with patch(
        "datasetinsights.stats.visualization.plots.add_single_bbox_on_image"
    ) as mock:
        plot_bboxes(img, boxes, label_mappings=label_mappings, colors=colors)
        assert mock.call_count == len(boxes)


@patch("datasetinsights.evaluation_metrics.confusion_matrix.Records")
def test_match_boxes(mock_record):
    match_results = [(0.5, True), (0.6, False), (0.7, True)]
    expected_colors = ["green", "red", "green"]
    mock_record.return_value.match_results.return_value = match_results
    colors = match_boxes(None, None)
    for i in range(len(colors)):
        assert colors[i] == expected_colors[i]


# @patch("datasetinsights.stats.visualization.bbox2d_plot._cv2.rectangle")
# def test__add_single_bbox_on_image(mock):
#     image = np.zeros((100, 200, 3))
#     left, top, right, bottom = 0, 0, 1, 1
#     color = "green"
#     box_line_width = 15
#     colors = [list(item) for item in _COLOR_NAME_TO_RGB[color]]
#     rgb_color, _ = colors
#     _add_single_bbox_on_image(
#         image,
#         left,
#         top,
#         right,
#         bottom,
#         label="car",
#         color=color,
#         box_line_width=box_line_width,
#     )
#     mock.assert_any_call(
#         image, (left, top), (right, bottom), rgb_color, box_line_width
#     )


@patch("datasetinsights.stats.visualization.bbox2d_plot._cv2.rectangle")
def test__add_single_bbox_on_image_edge_case(mock):
    image = np.zeros((100, 200, 3))
    left, top, right, bottom = 0, 150, 50, 200
    color = "green"
    box_line_width = 15
    colors = [list(item) for item in _COLOR_NAME_TO_RGB[color]]
    rgb_color, _ = colors
    _add_single_bbox_on_image(
        image,
        left,
        top,
        right,
        bottom,
        label="car",
        color=color,
        box_line_width=box_line_width,
    )
    mock.assert_any_call(
        image, (left, top), (right, bottom), rgb_color, box_line_width
    )


def test__add_single_bbox_on_image_throw_exception():
    image = np.zeros((100, 200, 3))
    with pytest.raises(TypeError):
        _add_single_bbox_on_image(
            image, "bad", "bad", "bad", "bad", label="car"
        )


@patch(
    "datasetinsights.stats.visualization.bbox2d_plot._add_single_bbox_on_image"
)
def test_add_single_bbox_on_image(mock):
    bbox = BBox2D(label=1, x=1, y=1, w=2, h=3)
    image = np.zeros((100, 200, 3))
    label = "car"
    color = "red"
    left, top = (bbox.x, bbox.y)
    right, bottom = (bbox.x + bbox.w, bbox.y + bbox.h)
    add_single_bbox_on_image(image, bbox, label, color)
    mock.assert_called_with(
        image,
        left,
        top,
        right,
        bottom,
        label=label,
        color=color,
        font_size=100,
        box_line_width=15,
    )
