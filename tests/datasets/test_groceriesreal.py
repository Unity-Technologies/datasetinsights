import os
import tempfile

from datasetinsights.datasets.groceries_real import GroceriesReal


def test_is_dataset_files_present_returns_true():
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "groceries_real_train.txt"), "x"):
            with open(os.path.join(tmp, "annotations.json"), "x"):
                assert GroceriesReal.is_dataset_files_present(tmp)


def test_is_dataset_files_present_returns_false():
    with tempfile.TemporaryDirectory() as tmp:
        assert not GroceriesReal.is_dataset_files_present(tmp)
