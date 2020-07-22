import dash_core_components as dcc
import dash_html_components as html

import datasetinsights.data.datasets.statistics as stat
import datasetinsights.visualization.constants as constants

from .plots import bar_plot, histogram_plot


def generate_total_counts_figure(max_samples, roinfo):
    """ Method for generating total object count bar plot using ploty.

    Args:
        max_samples(int): maximum number of samples that will be included
            in the plot.
        roinfo(datasetinsights.data.datasets.statistics.RenderedObjectInfo):
            Rendered Object Info in Captures.

    Returns:
        plotly.graph_objects.Figure: chart to display total object count
    """

    total_counts_fig = bar_plot(
        roinfo.total_counts(),
        x="label_id",
        y="count",
        x_title="Label Id",
        y_title="Count",
        title="Total Object Count in Dataset",
        hover_name="label_name",
    )
    return total_counts_fig


def generate_per_capture_count_figure(max_samples, roinfo):
    """ Method for generating object count per capture histogram using ploty.

    Args:
        max_samples(int): maximum number of samples that will be included
            in the plot.
        roinfo(datasetinsights.data.datasets.statistics.RenderedObjectInfo):
            Rendered Object Info in Captures.

    Returns:
        plotly.graph_objects.Figure: chart to display object counts per capture
    """

    per_capture_count_fig = histogram_plot(
        roinfo.per_capture_counts(),
        x="count",
        x_title="Object Counts Per Capture",
        y_title="Frequency",
        title="Distribution of Object Counts Per Capture",
        max_samples=max_samples,
    )
    return per_capture_count_fig


def generate_pixels_visible_per_object_figure(max_samples, roinfo):
    """ Method for generating pixels visible per object histogram using ploty.

    Args:
        max_samples(int): maximum number of samples that will be included
            in the plot.
        roinfo(datasetinsights.data.datasets.statistics.RenderedObjectInfo):
            Rendered Object Info in Captures.

    Returns:
        plotly.graph_objects.Figure: chart to display visible pixels per object
    """

    pixels_visible_per_object_fig = histogram_plot(
        roinfo.raw_table,
        x="visible_pixels",
        x_title="Visible Pixels Per Object",
        y_title="Frequency",
        title="Distribution of Visible Pixels Per Object",
        max_samples=max_samples,
    )

    return pixels_visible_per_object_fig


def overview(data_root):
    """ Method for displaying overview statistics.

    Args:
        data_root(str): path to the dataset.

    Returns:
        html layout: displays graphs for overview statistics.
    """

    roinfo = stat.RenderedObjectInfo(
        data_root=data_root, def_id=constants.RENDERED_OBJECT_INFO_DEFINITION_ID
    )

    total_counts_fig = generate_total_counts_figure(
        constants.MAX_SAMPLES, roinfo
    )
    per_capture_count_fig = generate_per_capture_count_figure(
        constants.MAX_SAMPLES, roinfo
    )
    pixels_visible_per_object_fig = generate_pixels_visible_per_object_figure(
        constants.MAX_SAMPLES, roinfo
    )

    overview_layout = html.Div(
        [
            html.Div(id="overview"),
            dcc.Graph(id="total_count", figure=total_counts_fig,),
            html.Div(
                [
                    dcc.Graph(
                        id="per_object_count", figure=per_capture_count_fig,
                    ),
                    dcc.Graph(
                        id="pixels_visible_per_object",
                        figure=pixels_visible_per_object_fig,
                    ),
                ],
                style={"columnCount": 2},
            ),
        ],
    )
    return overview_layout
