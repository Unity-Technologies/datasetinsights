import pathlib
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
from PIL import Image, ImageColor
from pytest import approx

from datasetinsights.datasets.cityscapes import CITYSCAPES_COLOR_MAPPING
from datasetinsights.io.bbox import BBox2D
from datasetinsights.stats.visualization.plots import (
    _convert_euler_rotations_to_scatter_points,
    bar_plot,
    decode_segmap,
    histogram_plot,
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
    boxes = [
        BBox2D(label="car", x=1, y=1, w=2, h=3),
        BBox2D(label="tree", x=7, y=6, w=3, h=4),
        BBox2D(label="light", x=2, y=6, w=2, h=4),
    ]
    colors = [
        ImageColor.getcolor("green", "RGB"),
        ImageColor.getcolor("red", "RGB"),
        ImageColor.getcolor("green", "RGB"),
    ]

    with patch(
        "datasetinsights.stats.visualization.plots.ImageDraw.Draw"
    ) as mock:
        instance = mock.return_value
        plot_bboxes(img, boxes, colors)
        assert instance.rectangle.call_count == len(boxes)
        assert instance.multiline_text.call_count == len(boxes)
