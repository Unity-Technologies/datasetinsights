import json
import tempfile
from pathlib import Path

from datasetinsights.datasets.transformers import (
    COCOInstancesTransformer,
    COCOKeypointsTransformer,
)


def assert_json_equals(file1, file2):
    with open(file1, "r") as f1:
        j1 = json.dumps(json.load(f1), sort_keys=True, indent=4)
    with open(file2, "r") as f2:
        j2 = json.dumps(json.load(f2), sort_keys=True, indent=4)

    assert j1 == j2


def test_coco_instances_transformer():
    parent_dir = Path(__file__).parent.parent.absolute()
    mock_data_dir = parent_dir / "mock_data" / "simrun"
    mock_coco_dir = parent_dir / "mock_data" / "coco"

    transformer = COCOInstancesTransformer(str(mock_data_dir))

    with tempfile.TemporaryDirectory() as tmp_dir:
        transformer.execute(tmp_dir)
        output_file = Path(tmp_dir) / "annotations" / "instances.json"
        expected_file = mock_coco_dir / "annotations" / "instances.json"
        output_image_folder = Path(tmp_dir) / "images"

        assert output_file.exists()
        assert output_image_folder.exists()
        assert list(output_image_folder.glob("*"))
        assert_json_equals(expected_file, output_file)


def test_coco_keypoints_transformer():
    parent_dir = Path(__file__).parent.parent.absolute()
    mock_data_dir = parent_dir / "mock_data" / "simrun_keypoint_dataset"
    mock_coco_dir = parent_dir / "mock_data" / "coco"

    transformer = COCOKeypointsTransformer(str(mock_data_dir))

    with tempfile.TemporaryDirectory() as tmp_dir:
        transformer.execute(tmp_dir)
        output_file = Path(tmp_dir) / "annotations" / "keypoints.json"
        expected_file = mock_coco_dir / "annotations" / "keypoints.json"
        output_image_folder = Path(tmp_dir) / "images"

        assert output_file.exists()
        assert output_image_folder.exists()
        assert list(output_image_folder.glob("*"))
        assert_json_equals(expected_file, output_file)
