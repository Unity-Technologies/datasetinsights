import collections
import json

import pytest

from datasetinsights.data.simulation import Captures
from datasetinsights.data.simulation.exceptions import DefinitionIDError
from datasetinsights.data.simulation.tables import SCHEMA_VERSION, glob


def test_filter_annotations(mock_data_dir):
    captures = Captures(str(mock_data_dir), version=SCHEMA_VERSION)

    annotation_counts = collections.defaultdict(int)
    json_files = glob(mock_data_dir, captures.FILE_PATTERN)
    for json_file in json_files:
        records = json.load(open(json_file, "r"))[Captures.TABLE_NAME]
        for record in records:
            for annotation in record["annotations"]:
                def_id = annotation["annotation_definition"]
                if annotation.get("values"):
                    annotation_counts[def_id] += len(annotation["values"])
                else:
                    annotation_counts[def_id] += 1
    for def_id, count in annotation_counts.items():
        assert len(captures.filter_annotations(def_id)) == count

    with pytest.raises(DefinitionIDError):
        captures.filter_annotations("bad_definition_id")


def test_normalize_annotations():
    data = {
        "id": "36db0", "annotation_definition": 1,
        "filename": None, "values": [{"a": 10, "b": 20}, {"a": 30, "b": 40}]
    }
    expected = [
        {
            "id": "36db0", "annotation_definition": 1,
            "filename": None, "values.a": 10, "values.b": 20
        },
        {
            "id": "36db0", "annotation_definition": 1,
            "filename": None, "values.a": 30, "values.b": 40
        },
    ]
    assert Captures._normalize_annotation(data) == expected


def test_normalize_annotations_empty_values():
    data = {
        "id": "36db0", "annotation_definition": 1,
        "filename": "abc.png", "values": None
    }
    assert Captures._normalize_annotation(data) == [data]
