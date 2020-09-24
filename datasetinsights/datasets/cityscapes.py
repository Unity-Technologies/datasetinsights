import logging
import os
from pathlib import Path

from torchvision import datasets

import datasetinsights.constants as const
from datasetinsights.io.gcs import GCSClient

from .base import Dataset

CITYSCAPES_GCS_PATH = "data/cityscapes"
CITYSCAPES_LOCAL_PATH = "cityscapes"
ZIPFILES = [
    "leftImg8bit_trainvaltest.zip",
    "gtFine_trainvaltest.zip",
]
CITYSCAPES_COLOR_MAPPING = {c.id: c.color for c in datasets.Cityscapes.classes}

logger = logging.getLogger(__name__)


class Cityscapes(Dataset):
    """
    Args:
        data_root (str): root directory prefix of all datasets
        split (str): indicate split type of the dataset (train|val|test)

    Attributes:
        root (str): root directory of the dataset
        split (str): indicate split type of the dataset (train|val|test)
    """

    def __init__(
        self,
        *,
        data_root=const.DEFAULT_DATA_ROOT,
        split="train",
        transforms=None,
        **kwargs,
    ):
        self.root = os.path.join(data_root, CITYSCAPES_LOCAL_PATH)
        self.split = split
        self.download(CITYSCAPES_GCS_PATH)

        self._cityscapes = datasets.Cityscapes(
            self.root,
            split=split,
            mode="fine",
            target_type="semantic",
            transforms=transforms,
        )

    def __getitem__(self, index):
        return self._cityscapes[index]

    def __len__(self):
        return len(self._cityscapes)

    def download(self, cloud_path):
        """Download cityscapes dataset

        Note:
            The current implementation assumes a GCS cloud path.
            Should we keep this method here if we want to support other cloud
            storage system?

        Args:
            cloud_path (str): cloud path of the dataset
        """
        path = Path(self.root)
        path.mkdir(parents=True, exist_ok=True)

        for zipfile in ZIPFILES:
            localfile = os.path.join(self.root, zipfile)
            if os.path.isfile(localfile):
                # TODO: Check file hash to verify file integrity
                logger.debug(f"File {localfile} exists. Skip download.")
                continue
            client = GCSClient()
            object_key = os.path.join(CITYSCAPES_GCS_PATH, zipfile)

            logger.debug(
                f"Downloading file {localfile} from gs://{const.GCS_BUCKET}/"
                f"{object_key}"
            )
            client.download(
                local_path=self.root, bucket=const.GCS_BUCKET, key=object_key
            )
