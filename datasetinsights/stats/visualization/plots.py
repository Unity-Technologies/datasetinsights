import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image, ImageColor

from datasetinsights.datasets.cityscapes import CITYSCAPES_COLOR_MAPPING
from datasetinsights.evaluation_metrics.confusion_matrix import (
    prediction_records,
)
from datasetinsights.stats.visualization.bbox2d_plot import (
    add_single_bbox_on_image,
)

logger = logging.getLogger(__name__)
COLORS = list(ImageColor.colormap.values())
FONT_SCALE = 35
LINE_WIDTH_SCALE = 250


def decode_segmap(labels, dataset="cityscapes"):
    """Decode segmentation class labels into a color image.

    Args:
        labels (np.array): an array of size (H, W) with integer grayscale
        values denoting the class label at each spatial location.
        dataset (str): dataset name. Defaults to "cityscapes".

    Returns:
        A np.array of the resulting decoded color image in (H, W, C).

    .. note:: (H, W, C) stands for the (height, width, channel) of the 2D image.
    """
    if dataset == "cityscapes":
        color_mapping = CITYSCAPES_COLOR_MAPPING
    else:
        raise ValueError(f"Dataset '{dataset}' is not supported.")

    h, w = labels.shape
    rgb = np.zeros((h, w, 3))
    for id, color in color_mapping.items():
        color = np.array(color)
        select_mask = labels == id
        rgb[select_mask, :] = color / 255.0

    return rgb


def grid_plot(images, figsize=(3, 5), img_type="rgb", titles=None):
    """ Plot 2D array of images in grid.
    Args:
        images (list): 2D array of images.
        figsize (tuple): target figure size of each image in the grid.
        Defaults to (3, 5).
        img_type (string): image plot type ("rgb", "gray"). Defaults to "rgb".
        titles (list[str]): a list of titles. Defaults to None.
    Returns:
        matplotlib figure the combined grid plot.
    """
    n_rows = len(images)
    n_cols = len(images[0])

    figsize = (figsize[0] * n_cols, figsize[1] * n_rows)

    figure = plt.figure(figsize=figsize, constrained_layout=True)
    for i in range(n_rows):
        for j in range(n_cols):
            img = images[i][j]
            k = i * n_cols + j + 1
            plt.subplot(n_rows, n_cols, k)
            plt.xticks([])
            plt.yticks([])
            if titles:
                plt.title(titles[k - 1])
            plt.grid(False)
            if img_type == "gray":
                plt.imshow(img, cmap="gray")
            else:
                plt.imshow(img, plt.cm.binary)
    plt.show()

    return figure


def _process_label(bbox, label_mappings=None):
    """Create a label text for the bbox.

    Args:
        bbox (BBox2D): a bounding box
        label_mappings (dict): a dict of {label_id: label_name} mapping
        Defaults to None.
    """
    if label_mappings is not None:
        label = label_mappings[bbox.label]
    if bbox.score != 1.0:
        return f"{label}: {bbox.score * 100: .2f}%"
    else:
        return label


def match_boxes(pred_bboxes, gt_bboxes):
    """ Provide a list of colors for pred annotations.

    Args:
        pred_bboxes (list[BBox2D]): a list of prediction bounding boxes
        gt_bboxes (list[BBox2D]): a list of ground truth bounding boxes

    Returns:
        list: a list of color names (either "green" or "red").
        green: if the predicted bounding box can be matched to a ground
        truth bounding box.
        red: if the predicted bounding box can't be matched to a ground
        truth bounding box.
    """
    records = prediction_records(gt_bboxes, pred_bboxes)
    match_results = records.match_results

    def get_color(match):
        if match:
            return "green"
        else:
            return "red"

    colors = [get_color(match) for _, match in match_results]
    return colors


def plot_bboxes(image, bboxes, label_mappings=None, colors=None):
    """ Plot an image with bounding boxes.

    For ground truth image, a color is randomly selected for each bounding box.
    For prediction, the color of a boundnig box is coded based on IOU value
    between prediction and ground truth bounding boxes. It is considered true
    positive if IOU >= 0.5. We only visualize prediction bounding box with
    score >= 0.5. For prediction, it's a green box if the predicted bounding box
    can be matched to a ground truth bounding boxes. It's a red box if the
    predicted bounding box can't be matched to a ground truth bounding boxes.

    Args:
        image (PIL Image): a PIL image.
        bboxes (list): a list of BBox2D objects.
        label_mappings (dict): a dict of {label_id: label_name} mapping
        Defaults to None.
        colors (list): a color list for boxes. Defaults to None.
        If colors = None, it will randomly assign PIL.COLORS for each box.

    Returns:
        PIL Image: a PIL image with bounding boxes drawn.
    """
    np_image = np.array(image)
    image_height, _, _ = np_image.shape
    for i, box in enumerate(bboxes):
        label = _process_label(box, label_mappings)
        color = colors[i] if colors else None
        font_size = image_height // FONT_SCALE
        box_line_width = image_height // LINE_WIDTH_SCALE
        add_single_bbox_on_image(
            np_image,
            box,
            label,
            color,
            font_size=font_size,
            box_line_width=box_line_width,
        )

    return Image.fromarray(np_image)


def bar_plot(
    df, x, y, title=None, x_title=None, y_title=None, x_tickangle=0, **kwargs
):
    """Create plotly bar plot

    Args:
        df (pd.DataFrame): A pandas dataframe that contain bar plot data.
        x (str): The column name of the data in x-axis.
        y (str): The column name of the data in y-axis.
        title (str, optional): The title of this plot.
        x_title (str, optional): The x-axis title.
        y_title (str, optional): The y-axis title.
        x_tickangle (int, optional): X-axis text tickangle (default: 0)

    This method can also take addition keyword arguments that can be passed to
    [plotly.express.bar](https://plotly.com/python-api-reference/generated/plotly.express.bar.html#plotly.express.bar) method.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"id": [0, 1, 2], "name": ["a", "b", "c"],
        ...                    "count": [10, 20, 30]})
        >>> bar_plot(df, x="id", y="count", hover_name="name")
    """  # noqa: E501 URL should not be broken down into lines
    fig = px.bar(df, x=x, y=y, **kwargs)
    fig = fig.update_layout(
        xaxis=dict(title=x_title, tickangle=x_tickangle),
        yaxis=dict(title=y_title),
        title_text=title,
    )

    return fig


def histogram_plot(
    df, x, max_samples=None, title=None, x_title=None, y_title=None, **kwargs
):
    """Create plotly histogram plot

    Args:
        df (pd.DataFrame): A pandas dataframe that contain raw data.
        x (str): The column name of the raw data for histogram plot.
        title (str, optional): The title of this plot.
        x_title (str, optional): The x-axis title.
        y_title (str, optional): The y-axis title.

    This method can also take addition keyword arguments that can be passed to
    [plotly.express.histogram](https://plotly.com/python-api-reference/generated/plotly.express.histogram.html) method.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"id": [0, 1, 2], "count": [10, 20, 30]})
        >>> histogram_plot(df, x="count")

        Histnorm plot using probability density:

        >>> histogram_plot(df, x="count", histnorm="probability density")
    """  # noqa: E501 URL should not be broken down into lines
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples)

    fig = px.histogram(df, x=x, **kwargs)
    fig = fig.update_layout(
        xaxis=dict(title=x_title), yaxis=dict(title=y_title), title_text=title,
    )

    return fig


def _convert_euler_rotations_to_scatter_points(
    df, x_col, y_col, z_col=None, max_samples=None
):
    """Turns euler rotations into a dataframe of points for plotting."""
    if max_samples is not None and len(df) > max_samples:
        df = df.sample(max_samples)

    for _, row in df.iterrows():
        v1 = np.array((0, 1, 0))

        theta_x = np.radians(row[x_col])

        rx = np.array(
            (
                (np.cos(theta_x), -np.sin(theta_x), 0),
                (np.sin(theta_x), np.cos(theta_x), 0),
                (0, 0, 1),
            )
        )
        v1 = rx.dot(v1)

        theta_y = np.radians(row[y_col])
        ry = np.array(
            (
                (1, 0, 0),
                (0, np.cos(theta_y), -np.sin(theta_y)),
                (0, np.sin(theta_y), np.cos(theta_y)),
            )
        )
        v1 = ry.dot(v1)

        if z_col is not None:
            theta_z = np.radians(row[z_col])
            rz = np.array(
                (
                    (np.cos(theta_z), 0, np.sin(theta_z)),
                    (0, 1, 0),
                    (-np.sin(theta_z), 0, np.cos(theta_z)),
                )
            )
            v1 = rz.dot(v1)

            text = """x: {x}°  y: {y}°  z: {z}°""".format(
                x=row[x_col], y=row[y_col], z=row[z_col]
            )
        else:
            text = """x: {x}°  y: {y}°""".format(x=row[x_col], y=row[y_col])

        ser = [v1[0], v1[1], v1[2], text]
        yield ser


def rotation_plot(df, x, y, z=None, max_samples=None, title=None, **kwargs):
    """Create a plotly 3d rotation plot
    Args:
        df (pd.DataFrame): A pandas dataframe that contains the raw data.
        x (str): The column name containing x rotations.
        y (str): The column name containing y rotations.
        z (str, optional): The column name containing z rotations.
        title (str, optional): The title of this plot.

    This method can also take addition keyword arguments that can be passed to
    [plotly.graph_objects.Scatter3d](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter3d.html) method.

    Returns:
        A plotly.graph_objects.Figure containing the scatter plot
    """  # noqa: E501 URL should not be broken down into lines
    rot_plot_columns = ("x", "y", "z", "text")
    dfrot = pd.DataFrame(
        list(
            _convert_euler_rotations_to_scatter_points(df, x, y, z, max_samples)
        ),
        columns=rot_plot_columns,
    )
    fig = (
        go.Figure(
            data=[
                go.Scatter3d(
                    x=dfrot["x"],
                    y=dfrot["y"],
                    z=dfrot["z"],
                    text=dfrot["text"],
                    mode="markers",
                    hoverinfo="text",
                    marker=dict(size=5, opacity=0.5),
                    **kwargs,
                )
            ]
        )
        .update_xaxes(showticklabels=False)
        .update_layout(
            title_text=title,
            scene=dict(
                xaxis=dict(showticklabels=False, range=[-1, 1]),
                yaxis=dict(showticklabels=False, range=[-1, 1]),
                zaxis=dict(showticklabels=False, range=[-1, 1]),
            ),
        )
    )
    return fig
