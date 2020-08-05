"""test captures."""
import collections
import json

import pytest

from datasetinsights.datasets.simulation import (
    SCHEMA_VERSION,
    Captures,
    DefinitionIDError,
    glob,
)


def test_get_captures_and_annotations(mock_data_dir):
    """test get captures and annotations."""
    captures = Captures(str(mock_data_dir), version=SCHEMA_VERSION)

    captures_per_definition = collections.defaultdict(int)
    json_files = glob(mock_data_dir, captures.FILE_PATTERN)
    for json_file in json_files:
        records = json.load(open(json_file, "r"))[Captures.TABLE_NAME]
        for record in records:
            for annotation in record["annotations"]:
                def_id = annotation["annotation_definition"]
                captures_per_definition[def_id] += 1

    for def_id, count in captures_per_definition.items():
        assert len(captures.filter(def_id)) == count

    with pytest.raises(DefinitionIDError):
        captures.filter("bad_definition_id")
