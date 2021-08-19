import json
import tempfile
from pathlib import Path

from datasetinsights.datasets.transformers import COCOTransformer


def assert_json_equals(file1, file2):
    j1 = json.dumps(json.loads(file1), sort_keys=True)
    j2 = json.dumps(json.loads(file2), sort_keys=True)
    assert j1 == j2


def test_coco_transformer(mock_data_dir):
    parent_dir = Path(__file__).parent.parent.absolute()
    mock_data_dir = parent_dir / "mock_data" / "simrun"
    mock_coco_dir = parent_dir / "mock_data" / "coco"
    transformer = COCOTransformer(str(mock_data_dir))

    with tempfile.TemporaryDirectory() as tmp_dir:
        transformer.execute(tmp_dir)
        output_file = Path(tmp_dir) / "instances.json"
        expected_file = mock_coco_dir / "instances.json"

        assert Path.exists(output_file)
        assert_json_equals(expected_file, output_file)
