import collections
import json

import pandas as pd
import pytest

from datasetinsights.datasets.simulation import (
    SCHEMA_VERSION,
    DefinitionIDError,
    Metrics,
    glob,
)


def test_filter_metrics(mock_data_dir):
    metrics = Metrics(str(mock_data_dir), version=SCHEMA_VERSION)

    expected_rows = collections.defaultdict(int)
    expected_cols = collections.defaultdict(set)
    exclude_metrics = set(["metric_definition", "values"])
    def_ids = set()
    actual_metrics = collections.defaultdict(pd.DataFrame)
    json_files = glob(mock_data_dir, metrics.FILE_PATTERN)
    for json_file in json_files:
        records = json.load(open(json_file, "r"))[Metrics.TABLE_NAME]
        for record in records:
            def_id = record["metric_definition"]
            def_ids.add(def_id)
            for key in record:
                if key not in exclude_metrics:
                    expected_cols[def_id].add(key)
            values = pd.json_normalize(record["values"])
            for key in values.columns:
                expected_cols[def_id].add(key)
            expected_rows[def_id] += len(values)

    for def_id in def_ids:
        actual_metrics[def_id] = metrics.filter_metrics(def_id)

    for def_id, expected_metric in actual_metrics.items():
        expected_shape = (expected_rows[def_id], len(expected_cols[def_id]))
        assert expected_shape == actual_metrics[def_id].shape
        assert expected_cols[def_id] == set(actual_metrics[def_id].columns)

    with pytest.raises(DefinitionIDError):
        metrics.filter_metrics("bad_definition_id")


def test_normalize_values(mock_data_dir):
    metrics = {
        "capture_id": "1234",
        "annotation_id": None,
        "sequence_id": "2345",
        "step": 50,
        "metric_definition": "193ce072-0e49-4ea4-a99f-7ca837e3a6ce",
        "values": [
            {
                "label_id": 1,
                "label_name": "book_dorkdiaries_aladdin",
                "count": 1,
            },
            {
                "label_id": 2,
                "label_name": "candy_minipralines_lindt",
                "count": 2,
            },
        ],
    }
    expected = [
        {
            "label_id": 1,
            "label_name": "book_dorkdiaries_aladdin",
            "count": 1,
            "capture_id": "1234",
            "annotation_id": None,
            "step": 50,
            "sequence_id": "2345",
        },
        {
            "label_id": 2,
            "label_name": "candy_minipralines_lindt",
            "count": 2,
            "capture_id": "1234",
            "annotation_id": None,
            "step": 50,
            "sequence_id": "2345",
        },
    ]
    flatten_metrics = Metrics._normalize_values(metrics)
    for i, metric in enumerate(expected):
        for k in metric:
            assert metric[k] == flatten_metrics[i][k]
