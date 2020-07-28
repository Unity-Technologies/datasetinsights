import argparse
import os

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import datasetinsights.data.datasets.statistics as stat
import datasetinsights.visualization.constants as constants
import datasetinsights.visualization.overview as overview
from datasetinsights.visualization.object_detection import (
    render_object_detection_layout,
)
from datasetinsights.visualization.plots import histogram_plot

this_dir = os.path.dirname(os.path.abspath(__file__))
css_file = os.path.join(this_dir, "stylesheet.css")

app = dash.Dash(__name__, external_stylesheets=[css_file],)
app.config.suppress_callback_exceptions = True


def main_layout(data_root):
    """ Method for generating main app layout.

    Args:
        data_root(str): path to the dataset.

    Returns:
        html layout: main layout design with tabs for overview statistics
            and object detection.
    """
    app_layout = html.Div(
        [
            html.H1(
                children="Dataset Insights",
                style={
                    "textAlign": "center",
                    "padding": 20,
                    "background": "lightgrey",
                },
            ),
            html.Div(
                [
                    dcc.Tabs(
                        id="page_tabs",
                        value="dataset_overview",
                        children=[
                            dcc.Tab(
                                label="Overview", value="dataset_overview",
                            ),
                            dcc.Tab(
                                label="Object Detection",
                                value="object_detection",
                            ),
                        ],
                    ),
                    html.Div(id="main_page_tabs"),
                ]
            ),
        ]
    )
    return app_layout


@app.callback(
    Output("main_page_tabs", "children"), [Input("page_tabs", "value")]
)
def render_content(tab):
    if tab == "dataset_overview":
        return overview.html_overview(data_root)
    elif tab == "object_detection":
        return render_object_detection_layout(data_root)


@app.callback(
    Output("pixels_visible_per_object", "figure"),
    [Input("pixels_visible_filter", "value")],
)
def update_visible_pixels_figure(label_value):
    roinfo = stat.RenderedObjectInfo(
        data_root=data_root, def_id=constants.RENDERED_OBJECT_INFO_DEFINITION_ID
    )
    filtered_roinfo = roinfo.raw_table[
        roinfo.raw_table["label_name"] == label_value
    ][["visible_pixels"]]
    filtered_figure = histogram_plot(
        filtered_roinfo,
        x="visible_pixels",
        x_title="Visible Pixels For " + str(label_value),
        y_title="Frequency",
        title="Distribution of Visible Pixels For " + str(label_value),
        max_samples=constants.MAX_SAMPLES,
    )
    return filtered_figure


@app.callback(
    Output("per_object_count", "figure"),
    [Input("object_count_filter", "value")],
)
def update_object_counts_capture_figure(label_value):
    roinfo = stat.RenderedObjectInfo(
        data_root=data_root, def_id=constants.RENDERED_OBJECT_INFO_DEFINITION_ID
    )
    filtered_object_count = roinfo.raw_table[
        roinfo.raw_table["label_name"] == label_value
    ]
    filtered_object_count = (
        filtered_object_count.groupby(["capture_id"])
        .size()
        .to_frame(name="count")
        .reset_index()
    )
    filtered_figure = histogram_plot(
        filtered_object_count,
        x="count",
        x_title="Object Counts Per Capture For " + str(label_value),
        y_title="Frequency",
        title="Distribution of Object Counts Per Capture For "
        + str(label_value),
        max_samples=constants.MAX_SAMPLES,
    )
    return filtered_figure


def check_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise ValueError(f"Path {path} not found")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", help="Path to the data root")
    args = parser.parse_args()
    data_root = check_path(args.data_root)
    app.layout = main_layout(data_root)
    app.run_server(debug=True)
