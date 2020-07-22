import pytest

import datasetinsights.visualization.constants as constants
from datasetinsights.data.simulation import Captures
from datasetinsights.data.simulation.tables import SCHEMA_VERSION
from datasetinsights.visualization.object_detection import ScaleFactor
from datasetinsights.visualization.plots import histogram_plot


def test_read_scale(mock_data_dir):
    mock_sensor = {
        "sensor_id": "9c5847b2-fb17-4c17-8cce-dc9f1d0700b1",
        "ego_id": "3649de42-454d-48a0-a596-aff9af4af46d",
        "modality": "camera",
        "translation": [0.0, 0.0, 0.0],
        "rotation": [0.0, 0.0, 0.0, 1.0],
        "scale": 1.0,
    }
    scale_factor = ScaleFactor(str(mock_data_dir))
    actual_scale = scale_factor._read_scale(str(mock_sensor))
    expected_scale = 1.0
    assert expected_scale == actual_scale


def test_generate_scale_factor_figures(mock_data_dir):
    captures = Captures(str(mock_data_dir), version=SCHEMA_VERSION)
    scale_factor = ScaleFactor(str(mock_data_dir))
    actual_figure = scale_factor._generate_scale_factor_figures(captures)

    captures["scale"] = captures["sensor"].apply(scale_factor._read_scale)

    expected_figure = histogram_plot(
        captures,
        x="scale",
        x_title="Scale",
        y_title="Capture count",
        title="Distribution of Scale Factor",
        range_x=[min(captures["scale"]), max(captures["scale"])],
        max_samples=constants.MAX_SAMPLES,
    )
    assert expected_figure == actual_figure
