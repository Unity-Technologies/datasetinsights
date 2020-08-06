import json

from datasetinsights.datasets.simulation import (
    SCHEMA_VERSION,
    AnnotationDefinitions,
    MetricDefinitions,
    glob,
)


def test_annotation_definitions(mock_data_dir):
    definition = AnnotationDefinitions(
        str(mock_data_dir), version=SCHEMA_VERSION
    )

    json_file = next(glob(mock_data_dir, AnnotationDefinitions.FILE_PATTERN))
    records = json.load(open(json_file, "r"))[AnnotationDefinitions.TABLE_NAME]

    def_ids = [r["id"] for r in records]
    for (i, def_id) in enumerate(def_ids):
        record = records[i]

        assert definition.get_definition(def_id) == record


def test_metric_definitions(mock_data_dir):
    definition = MetricDefinitions(str(mock_data_dir), version=SCHEMA_VERSION)

    json_file = next(glob(mock_data_dir, MetricDefinitions.FILE_PATTERN))
    records = json.load(open(json_file, "r"))[MetricDefinitions.TABLE_NAME]

    def_ids = [r["id"] for r in records]
    for (i, def_id) in enumerate(def_ids):
        record = records[i]

        assert definition.get_definition(def_id) == record
