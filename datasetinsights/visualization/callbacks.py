import datasetinsights.data.datasets.statistics as stat
import datasetinsights.visualization.constants as constants
import __main__

from datasetinsights.visualization.plots import histogram_plot
from dash.dependencies import Input, Output
from datasetinsights.visualization.app import app


# @app.callback(
#     Output("pixels_visible_filter_graph", "figure"),
#     [Input("pixels_visible_filter", "value")],
# )
# def update_visible_pixels_figure(label_value):
#     roinfo = stat.RenderedObjectInfo(
#         data_root=__main__.data_root,
#         def_id=constants.RENDERED_OBJECT_INFO_DEFINITION_ID
#     )
#     filtered_roinfo = roinfo.raw_table[
#         roinfo.raw_table["label_name"] == label_value
#     ][["visible_pixels"]]
#     filtered_figure = histogram_plot(
#         filtered_roinfo,
#         x="visible_pixels",
#         x_title="Visible Pixels For " + str(label_value),
#         y_title="Frequency",
#         title="Distribution of Visible Pixels For " + str(label_value),
#         max_samples=constants.MAX_SAMPLES,
#     )
#     return filtered_figure


@app.callback(
    Output("per_object_count_filter_graph", "figure"),
    [Input("object_count_filter", "value")],
)
def update_object_counts_capture_figure(label_value):
    roinfo = stat.RenderedObjectInfo(
        data_root=__main__.data_root,
        def_id=constants.RENDERED_OBJECT_INFO_DEFINITION_ID
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
