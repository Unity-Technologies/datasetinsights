import os
import tempfile

from datasetinsights.datasets.groceries_real import GroceriesReal


def test_is_groceries_real_dataset_files_present_returns_true():
    with tempfile.TemporaryDirectory() as tmp:
        temp_images_dir = os.path.join(tmp, "images")
        os.mkdir(temp_images_dir)
        open(os.path.join(temp_images_dir, "IMG_4185.JPG"), "x")
        open(os.path.join(tmp, "groceries_real_train.txt"), "x")
        open(os.path.join(tmp, "annotations.json"), "x")

        assert GroceriesReal.is_groceries_real_dataset_files_present(tmp)


def test_is_groceries_real_dataset_files_present_returns_false():
    with tempfile.TemporaryDirectory() as tmp:
        assert not GroceriesReal.is_groceries_real_dataset_files_present(tmp)
