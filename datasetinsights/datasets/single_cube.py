import logging
import os
from pathlib import Path
import zipfile
import json
import pandas as pd
import numpy as np
import torch

from PIL import Image

from .base import Dataset
import datasetinsights.constants as const
from datasetinsights.storage.gcs import GCSClient


VGG_SLAM_GCS_PATH = "data/single_cube/"
VGG_SLAM_LOCAL_PATH = "single_cube"
GCS_BUCKET = "thea-dev"
LOCAL_UNZIP_LOGS = "single_cube/Logs"
LOCAL_UNZIP_IMAGES = "single_cube/ScreenCapture"

logger = logging.getLogger(__name__)


class SingleCube(Dataset):
    """Single Cube dataset

    The metric is defined for grayscale depth images.

    Attributes:
        data_root (str): path towards the data
        split (str): split the data but it depends if we are in a training or
        in a test mode
        version (str): version of the dataset. There are two version: small
        and large
    """

    def __init__(
        self,
        *,
        config,
        data_root=const.DEFAULT_DATA_ROOT,
        split="train",
        version=None,
        transforms=None,
        **kwargs,
    ):
        self.config = config
        self.root = os.path.join(data_root, VGG_SLAM_LOCAL_PATH)
        self.data_root = data_root
        self.split = split
        self.version = version
        self.download()
        self.vgg_slam_data = self._get_annotations()
        self.transforms = transforms

    def __getitem__(self, index):
        """Abstract method to train estimators
        """

        df_data = self.vgg_slam_data
        image_name = df_data.screenCaptureName[index]

        target = df_data[['q_w', 'q_x', 'q_y', 'q_z']].loc[index].values
        # target = df_data[['x', 'y', 'z']].loc[index].values

        image = Image.open(
            os.path.join(self.data_root, LOCAL_UNZIP_IMAGES, image_name))

        # if self.config.system.distributed:
        if self.transforms:
            image = self.transforms(image).unsqueeze(0)
            target = np.asarray(target)
            target = torch.tensor(target, dtype=torch.float)

        return image, target

    def _get_indexes(self):
        df_data = self.vgg_slam_data
        list_index = list(df_data.index.values)
        return list_index

    def __len__(self):
        """Abstract method to have the number of rows of the dataset
        """

        return self.vgg_slam_data.shape[0]

    def _get_local_data_zip(self):
        """create a local path for download zip file
        """

        return os.path.join(self.root, f"{self.version}.zip")

    def _get_annotations(self):
        """Read data from the list of log.txt.
        In the text file, image name and target co-ordinates are saved in a
        list of json.
        For example:
            [
                {"x":-0.29225119948387148,
                "y":0.050000011920928958,
                "z":-0.20810827612876893,
                "screenCaptureName":"image_78"},
            ]

        Return:
            data frame of [x,y,z,image] columns
        """
        file_path = os.path.join(self.data_root, LOCAL_UNZIP_LOGS)
        files = os.listdir(file_path)
        keyword = 'DataCapture_'
        y_list = []
        for file in files:
            if keyword in file:
                with open(file_path + "/" + file) as file_in:
                    y_list1 = []
                    for line in file_in:
                        y_list.append(json.loads(line.split("\n")[0]))
                y_list += y_list1
        df_data = pd.DataFrame.from_dict(y_list, orient='columns')
        df_data['screenCaptureName'] = df_data['screenCaptureName'].apply(
            lambda x: x + '.png')

        return df_data

    def download(self):
        """Abstract method to download dataset from GCS
        """

        path = Path(self.root)
        path.mkdir(parents=True, exist_ok=True)
        client = GCSClient()
        object_key = os.path.join(VGG_SLAM_GCS_PATH, f"{self.version}.zip")
        data_zip_local = self._get_local_data_zip()
        if not os.path.exists(data_zip_local):
            logger.info(f"no data zip file found, will download.")
            client.download(
                bucket_name=GCS_BUCKET,
                object_key=object_key,
                localfile=data_zip_local,
            )

            with zipfile.ZipFile(data_zip_local, "r") as zip_dir:
                zip_dir.extractall(f"{self.root}")
