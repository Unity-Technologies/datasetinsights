import json
import tempfile
from pathlib import Path

import pytest

from datasetinsights.datasets.unity_perception import (
    AnnotationDefinitions,
    MetricDefinitions,
)
from datasetinsights.datasets.unity_perception.tables import (
    SCHEMA_VERSION,
    glob,
)
from datasetinsights.datasets.unity_perception.validation import (
    DuplicateRecordError,
    NoRecordError,
)


def test_annotation_definitions(mock_data_dir):
    definition = AnnotationDefinitions(
        str(mock_data_dir), version=SCHEMA_VERSION
    )

    json_file = next(glob(mock_data_dir, AnnotationDefinitions.FILE_PATTERN))
    records = json.load(open(json_file, "r", encoding="utf8"))[
        AnnotationDefinitions.TABLE_NAME
    ]

    def_ids = [r["id"] for r in records]
    for (i, def_id) in enumerate(def_ids):
        record = records[i]

        assert definition.get_definition(def_id) == record


def test_annotation_definitions_find_by_name():
    def1 = {
        "id": 1,
        "name": "good name",
        "description": "does not matter",
        "format": "JSON",
        "spec": [],
    }
    def2 = {
        "id": 2,
        "name": "another good name",
        "description": "does not matter",
        "format": "JSON",
        "spec": [],
    }
    ann_def = {
        "version": SCHEMA_VERSION,
        "annotation_definitions": [def1, def2],
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        with open(Path(tmp_dir) / "annotation_definitions.json", "w") as f:
            json.dump(ann_def, f)
        definition = AnnotationDefinitions(tmp_dir, version=SCHEMA_VERSION)

    pattern = r"^good\sname$"
    assert definition.find_by_name(pattern) == def1

    pattern = "good name"
    with pytest.raises(DuplicateRecordError):
        definition.find_by_name(pattern)

    pattern = "w;fhohfoewh"
    with pytest.raises(NoRecordError):
        definition.find_by_name(pattern)


def test_metric_definitions(mock_data_dir):
    definition = MetricDefinitions(str(mock_data_dir), version=SCHEMA_VERSION)

    json_file = next(glob(mock_data_dir, MetricDefinitions.FILE_PATTERN))
    records = json.load(open(json_file, "r"))[MetricDefinitions.TABLE_NAME]

    def_ids = [r["id"] for r in records]
    for (i, def_id) in enumerate(def_ids):
        record = records[i]

        assert definition.get_definition(def_id) == record
