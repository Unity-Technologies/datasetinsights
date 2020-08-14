import collections
import json

import pytest

from pathlib import Path

from datasetinsights.datasets.unity_perception import Captures
from datasetinsights.datasets.unity_perception.exceptions import (
    DefinitionIDError,
)
from datasetinsights.datasets.unity_perception.tables import (
    SCHEMA_VERSION,
    glob,
)


@pytest.mark.parametrize(
    "data_dir_name", ["simrun", "no_annotations_or_metrics"],
)
def test_get_captures_and_annotations(mock_data_base_dir, data_dir_name):
    mock_data_dir = mock_data_base_dir / data_dir_name
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
