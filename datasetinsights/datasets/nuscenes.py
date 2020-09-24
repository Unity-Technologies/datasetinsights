import logging
import os
import tarfile
import tempfile
from typing import Dict, List

from nuscenes.eval.detection import loaders
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from torch.utils.data import Dataset

import datasetinsights.constants as const
from datasetinsights.io.bbox import BBox3d
from datasetinsights.io.gcs import GCSClient

logger = logging.getLogger(__name__)

NUSCENES_GCS_PATH = "data/nuscenes"
NUSCENES_LOCAL_PATH = "sets/nuscenes"
CAMERA_KEYS = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
]
RADAR_KEYS = [
    "RADAR_FRONT",
    "RADAR_FRONT_LEFT",
    "RADAR_FRONT_RIGHT",
    "RADAR_BACK_LEFT",
    "RADAR_BACK_RIGHT",
]
LIDAR_KEY = "LIDAR_TOP"
SENSOR_DATA_KEYS = CAMERA_KEYS + RADAR_KEYS + [LIDAR_KEY]
GLOBAL = "global"
SENSOR = "sensor"
COORDINATES = [GLOBAL, SENSOR]
NUM_TRAINVAL_TARBALLS = 10
VERSIONS = ["v1.0"]
SPLITS = ["train", "val"]


# TODO: This should be a child class of abstract Dataset class
class NuscenesDataLoader(Dataset):
    """ Nuscenes Dataset https://www.nuscenes.org/
    Args:
        data_root (str): root directory prefix of all datasets
        split: test or valuation. see SPLITS constant for possible values
        version: numerical version of nuscenes dataset.
        sensor_data: what sensor data to use, can be any of the strings in
        SENSOR_DATA_KEYS
        full_dataset: whether to use the full dataset. If false, use the mini
        dataset
        download: whether or no to download the nuscenes dataset from gcs. If
        false, it is assumed the
        dataset is stored at the location specified by root
        coordinates: whether to use coordinates in the local coordinate system
        of the sensor or the global
        log coordinates. Options: 'sensor' or 'global'

    Attributes:
        root: directory where nuscenes data is stored. If no value is provided,
            the specified version of nuscenes is downloaded.
    """

    def __init__(
        self,
        *,
        data_root=const.DEFAULT_DATA_ROOT,
        split="train",
        version="v1.0",
        full_dataset: bool = False,
        download=False,
        sensor_data: List[str] = SENSOR_DATA_KEYS,
        coordinates="global",
    ):
        self.root = os.path.join(data_root, NUSCENES_LOCAL_PATH)
        if version not in VERSIONS:
            raise ValueError(
                f"version provided was {version} but only valid versions are: "
                f"{VERSIONS}"
            )
        if full_dataset is False:
            split = f"mini_{split}"
            version = f"{version}-mini"
        else:
            version = f"{version}-trainval"
        if download:
            self.download(version=version)
        nu = NuScenes(dataroot=self.root, version=version)
        self.nu = nu
        self.scenes = nu.scene
        self.split = split
        self.sample_tokens = loaders.load_gt(
            nusc=nu, eval_split=split
        ).sample_tokens
        for s in sensor_data:
            if s not in SENSOR_DATA_KEYS:
                raise ValueError(
                    f"sensor key: {s} is not a valid sensor. "
                    f"Valid sensors are: {SENSOR_DATA_KEYS}"
                )
        self.data_keys = sensor_data
        if coordinates not in COORDINATES:
            raise ValueError(
                f"coordinates can only be one of {COORDINATES} "
                f"but {self.coordinates} was given."
            )
        self.coordinates = coordinates

    def _merge_data_blobs(self, blob, canonical_dir):
        """Merge the data blobs together which comprise the v1.0-trainval
        dataset.
        The structure of each data blob in the nuscenes dataset is
        -v1.0-trainval01_blobs
         |-samples
           |-CAM_FRONT
           |-...all sensors
         |-sweeps
           |-CAM_FRONT
           |-...all sensors
        we need to merge all the contents into one single canonical directory
         which will have the format
        -canonical_dir
         |-samples
           |-CAM_FRONT
           |-...all sensors
         |-sweeps
           |-CAM_FRONT
           |-...all sensors
        Args:
            blob: filename of data blob
            canonical_dir: filename of directory to store all samples find in
            all datablobs
        """
        for d in ["samples", "sweeps"]:
            for sub_dir in SENSOR_DATA_KEYS:
                data_type_dir = os.path.join(blob, d, sub_dir)
                for name in os.listdir(data_type_dir):
                    src = os.path.join(data_type_dir, name)
                    dest = os.path.join(canonical_dir, d, sub_dir, name)
                    os.rename(src, dest)

    def _download_trainval_tars(self) -> (str, List[str]):
        """

        Returns:
            Tuple: First item is the path to the meta data dir, and the second
             item is a list of paths to the
            tarballs comprising the nuscenes trainval dataset
        """
        data_tarballs = [
            f"{NUSCENES_GCS_PATH}/v1.0-trainval0{1 + i}_blobs.tar"
            for i in range(NUM_TRAINVAL_TARBALLS - 1)
        ]
        data_tarballs.append(f"{NUSCENES_GCS_PATH}/v1.0-trainval10_blobs.tar")
        trainval_tars = []
        for data in data_tarballs:
            local_dest = os.path.join(self.root, os.path.basename(data))
            trainval_tars.append(local_dest)
            self.cloud_client.download(
                local_path=self.root, bucket=const.GCS_BUCKET, key=data
            )
        meta_gcs_key = f"{NUSCENES_GCS_PATH}/v1.0-trainval_meta.tar"
        meta_local_path = os.path.join(self.root, "v1.0-trainval_meta.tar")
        self.cloud_client.download(
            local_path=self.root, bucket=const.GCS_BUCKET, key=meta_gcs_key,
        )
        return meta_local_path, trainval_tars

    def _download_mini(self):
        mini_tar_path = f"{NUSCENES_GCS_PATH}/v1.0-mini.tar"
        local_tar_path = f"{self.root}/v1.0-mini.tar"
        logger.info("downloading mini dataset from gcs")
        self.cloud_client.download(
            local_path=self.root, bucket=const.GCS_BUCKET, key=mini_tar_path,
        )
        with tarfile.open(local_tar_path) as t:
            t.extractall(self.root)
        return self.root

    def download(self, version="v1.0-mini"):
        """
        download the nuscenes dataset version specified to /data/sets/nuscenes
        """
        self.cloud_client = GCSClient()
        if version == "v1.0-mini":
            return self._download_mini()
        elif version == "v1.0-trainval":
            meta_local_path, trainval_tars = self._download_trainval_tars()
            with tarfile.open(meta_local_path) as t:
                t.extractall(self.root)
            first_data_blob = trainval_tars[0]
            with tarfile.open(first_data_blob) as t:
                logger.info(f"extracting files from {first_data_blob}")
                t.extractall(self.root)
            os.remove(trainval_tars[0])
            for tar_path in trainval_tars[1:]:
                with tarfile.open(tar_path) as t:
                    logger.info(f"extracting files from {tar_path}")
                    with tempfile.TemporaryDirectory() as tmpdirname:
                        t.extractall(tmpdirname)
                        self._merge_data_blobs(
                            blob=tmpdirname, canonical_dir=self.root
                        )
            return self.root
        else:
            raise ValueError()

    def _nu_box2bbox3d(self, nu_box: Box, idx) -> BBox3d:
        width, length, height = nu_box.wlh
        return BBox3d(
            rotation=nu_box.orientation,
            translation=nu_box.center,
            label=nu_box.label,
            score=nu_box.score,
            velocity=nu_box.velocity,
            size=[width, height, length],
            sample_token=idx,
        )

    def __getitem__(self, item) -> Dict[str, List[BBox3d]]:
        """
        Args:
            item: index
        Returns:
            Dictionary: mapping the name of each sensor to the list of bounding
             boxes found in that sensor's data

        """
        data = {}
        sample_token = self.sample_tokens[item]
        sample_timeframe = self.nu.get("sample", sample_token)
        if self.coordinates == SENSOR:
            for sensor in self.data_keys:
                sensor_token = sample_timeframe["data"][sensor]
                data_path, nu_boxes, camera = self.nu.get_sample_data(
                    sensor_token
                )
                bboxes = [self._nu_box2bbox3d(b, item) for b in nu_boxes]
                data[sensor] = bboxes
        elif self.coordinates == GLOBAL:
            for sensor in self.data_keys:
                sensor_token = sample_timeframe["data"][sensor]
                nu_boxes = self.nu.get_boxes(sensor_token)
                bboxes = [self._nu_box2bbox3d(b, item) for b in nu_boxes]
                data[sensor] = bboxes
        else:
            raise ValueError(
                f"coordinates can only be one of {COORDINATES} but "
                f"{self.coordinates} was given."
            )
        return data

    def __len__(self):
        return len(self.sample_tokens)
