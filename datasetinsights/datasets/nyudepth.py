"""Load NYU Depth V2 data.

Downloaded NYUDepth dataset link:
    https://drive.google.com/drive/folders/1TzwfNA5JRFTPO-kHMU___kILmOEodoBo
This dataset is not the official NYU Depth V2 datset.
The data are preprocessed.
For more information, see the link below:
    https://github.com/ialhashim/DenseDepth
"""
import logging
import os
from zipfile import ZipFile

import pandas as pd
from PIL import Image

import datasetinsights.constants as const
from datasetinsights.io.gcs import GCSClient

from .base import Dataset

NYU_GCS_PATH = "data/nyudepth"
NYUDEPTH_LOCAL_PATH = "nyudepth"
SPLITS = ["train", "val", "test"]
ZIPFILE = "nyu_v2_subset_data.zip"
UNZIP_NAME = "data"
REAL_IMAGE_HEADER = "real"
TARGET_IMAGE_HEADER = "target"

logger = logging.getLogger(__name__)


class NyuDepth(Dataset):
    """
    Attributes:
        root (str): local directory where nyu_v2 dataset is saved
        nyu2_data (Pandas DataFrame): training or testing data
        split (str): test or valuation. see SPLITS constant for possible values
    """

    def __init__(
        self,
        data_root=const.DEFAULT_DATA_ROOT,
        transforms=None,
        split="train",
        cloud_path=NYU_GCS_PATH,
    ):
        """
        Args:
            data_root (str): root directory prefix of all datasets
            split (str): indicate split type of the dataset (train|test)
            cloud_path (str): cloud storage path where a copy of nyu_v2 is
            saved
        """
        self.root = os.path.join(data_root, NYUDEPTH_LOCAL_PATH)
        self.transforms = transforms
        self.split = split
        if not os.path.isdir(self.root):
            os.mkdir(self.root)
        if split not in SPLITS:
            raise ValueError(
                f"invalid value for split: {split}, possible values "
                f"are: {SPLITS}"
            )
        self.download(cloud_path=cloud_path)

        unzip_dir = os.path.join(self.root, UNZIP_NAME)

        self.nyu2_data = pd.read_csv(
            os.path.join(unzip_dir, f"nyu2_{split}.csv"),
            header=None,
            names=[REAL_IMAGE_HEADER, TARGET_IMAGE_HEADER],
            skip_blank_lines=True,
        )

    def __getitem__(self, index):
        """
        Args:
            index (int): index of the element
        Returns:
            sample (tuple): (real image, depth target)
        """
        real_img_path = self.nyu2_data[REAL_IMAGE_HEADER][index]
        target_img_path = self.nyu2_data[TARGET_IMAGE_HEADER][index]
        real_img = Image.open(os.path.join(self.root, real_img_path))
        target_img = Image.open(os.path.join(self.root, target_img_path))
        sample = (real_img, target_img)
        if self.transforms:
            sample = self.transforms(sample)
        return sample

    def __len__(self):
        return len(self.nyu2_data)

    def download(self, cloud_path):
        """Download nyu_v2 dataset
        The directory structure of the downloaded data is
        |--self.root
           |--nyudepth
               |--nyu_data.zip
               |--data
                   |--nyu2_test.csv
                   |--nyu2_test
                         |--00000_colors.png
                         |--00000_depth.png ...
                         |--01448_colors.png
                         |--01448_depth.png
                   |--nyu2_train.csv
                   |--nyu2_train
                         |--basement_0001a_out
                              |--1.jpg
                              |--1.png ...
                              |--281.jpg
                              |--281.png
                         ...
                         |--study_room_0005b_out
                              |--1.jpg
                              |--1.png ...
                              |--133.jpg
                              |--133.png
        Args:
            cloud_path (str): cloud path of the dataset
        """
        zip_file = os.path.join(self.root, ZIPFILE)
        unzip_dir = os.path.join(self.root, UNZIP_NAME)

        if os.path.isfile(zip_file):
            logger.debug(f"File {zip_file} exists. Skip download.")
        else:
            client = GCSClient()
            object_key = os.path.join(NYU_GCS_PATH, ZIPFILE)

            logger.debug(
                f"Downloading file {zip_file} from gs://{const.GCS_BUCKET}/"
                f"{object_key}"
            )
            client.download(
                local_path=self.root, bucket=const.GCS_BUCKET, key=object_key,
            )

        if os.path.isdir(unzip_dir):
            logger.debug(f"File {unzip_dir} exists. Skip unzip.")
        else:
            # unzip the file
            with ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(self.root)
                logger.debug(f"Unzip file from {zip_file}")
