import logging
import os
import shutil
import tempfile
import zipfile

import numpy as np
from PIL import Image
from pyquaternion import Quaternion

import datasetinsights.constants as const
from datasetinsights.io.bbox import BBox3d
from datasetinsights.io.gcs import GCSClient

from .base import Dataset
from .nuscenes import Box

logger = logging.getLogger(__name__)
KITTI_GCS_PATH = "data/kitti"
SPLITS = ["train", "test", "val", "trainval"]  # test refers to KITTI's test
# set which doesn't have labels

KITTI = "kitti"
NUSCENES = "nuscenes"
VALID_FORMATS = [KITTI, NUSCENES]
SAMPLEX_INDICES_FILE = "samples.txt"
ZIP_FILES = [
    "data_object_calib.zip",
    "data_object_image_2.zip",
    "data_object_label_2.zip",
]


class KittiBox3d:
    """
    class to represent a bounding box for the kitti dataset. Note that this
    style of bounding box is not primarily supported. The canonical 3d bounding
    box class for this repo is the class BBox3D.
    Reference code for KittiBox3d found at
    http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d
    https://github.com/bostondiditeam/kitti/tree/master/resources/devkit_object
    To convert from kitti style to nuscenes use the method convert_kitti2nu.
    """

    def __init__(
        self,
        *,
        label: str,
        position,
        dimensions,
        angle,
        sample_idx,
        score: float = 1.0,
    ):
        """

        Args:
            label: class label possibilities are 'car' 'pedestrian' and
            'cyclist'
            position: x,y,z in camera coordinates (meters)
            dimensions: length, height, width  (meters)
            angle: angle about vertical axis in radians range [-pi, pi]
            sample_idx: the index corresponding to the sample data (image), the
             image is stored in
            image_2/{sample_index:06d}.png
            score: confidence score (defaults to zero, to be used for ground
            truth)
        """
        self.label = label
        self.position = position
        self.dimensions = dimensions
        self.angle = angle
        self.score = score
        self.sample_idx = sample_idx


# todo add tests
class KittiTransforms:
    """
    Class to hold transformation matrices for a kitti data sample see more at
    https://github.com/yanii/kitti-pcl/blob/master/KITTI_README.TXT
    """

    def __init__(self, calib_filename):
        self.lines = [line.rstrip() for line in open(calib_filename)]

    def _get_velo2camera(self):
        """
        matrix takes a point in Velodyne coordinates and transforms it into the
        coordinate system of the left video camera. Likewise it serves as a
        representation of the Velodyne coordinate frame in camera coordinates.
        Returns: Combined translation and rotation matrix

        """
        velo_to_cam = np.array(
            self.lines[5].strip().split(" ")[1:], dtype=np.float32
        )
        velo_to_cam.resize((3, 4))
        return velo_to_cam

    @property
    def velo_to_cam_rotation(self):
        """
        Rotation matrix takes a point in Velodyne coordinates and transforms it
         into the
        coordinate system of the left video camera. Likewise it serves as a
        representation of the Velodyne coordinate frame in camera coordinates.
        Returns: Rotation matrix

        """
        velo_to_cam = self._get_velo2camera()
        return velo_to_cam[:, :3]

    @property
    def velo_to_cam_translation(self):
        """
        Translation matrix takes a point in Velodyne coordinates and transforms
         it into the
        coordinate system of the left video camera. Likewise it serves as a
        representation of the Velodyne coordinate frame in camera coordinates.
        Returns: Translation matrix

        """
        velo_to_cam = self._get_velo2camera()
        return velo_to_cam[:, 3]

    @property
    def r0_rect(self) -> np.ndarray:
        """

        Returns: Quaternion to rectify camera frame.

        """
        r0_rect = np.array(
            self.lines[4].strip().split(" ")[1:], dtype=np.float32
        )
        r0_rect.resize((3, 3))
        return r0_rect

    @property
    def projection_mat_left(self):
        """

        Returns: Projection matrix for left image (to project bounding box
        coordinate to image coordinates).

        """
        p_left = np.array(
            self.lines[2].strip().split(" ")[1:], dtype=np.float32
        )
        return p_left.resize((3, 4))

    @property
    def projection_rect_combined(self):
        """
        Merge rectification and projection into one matrix.

        Returns: combined rectification and projection matrix

        """
        p_combined = np.eye(4)
        p_combined[:3, :3] = self.r0_rect
        p_combined = np.dot(self.projection_mat_left, p_combined)
        return p_combined


def convert_kitti2nu(
    *, bbox: KittiBox3d, transforms: KittiTransforms
) -> BBox3d:
    """
    convert a bounding box from kitti format to nuscenes format

    Args:
        bbox: bounding box in kitti format
        transforms: camera transforms

    Returns:

    """
    center = bbox.position
    wlh = [
        bbox.dimensions[2],
        bbox.dimensions[0],
        bbox.dimensions[1],
    ]  # lhw -> wlh bbox['wlh']
    yaw_camera = bbox.angle
    name = bbox.label
    score = bbox.score

    # The Box class coord system is oriented the same way as as KITTI LIDAR: x
    # forward, y left, z up.
    # For rotation confer: http://www.cvlibs.net/datasets/kitti/setup.php.

    # 1: Create box in Box coordinate system with center at origin.
    # The second rotation in yaw_box transforms the coordinate frame from the
    # object frame
    # to KITTI camera frame. The equivalent cannot be naively done afterwards,
    # as it's box rotation
    # around the local object coordinate frame, rather than the camera frame.
    quat_box = Quaternion(axis=(0, 1, 0), angle=yaw_camera) * Quaternion(
        axis=(1, 0, 0), angle=np.pi / 2
    )
    box = Box([0.0, 0.0, 0.0], wlh, quat_box, name=name)

    # 2: Translate: KITTI defines the box center as the bottom center of the
    # vehicle. We use true center,
    # so we need to add half height in negative y direction, (since y points
    # downwards), to adjust. The
    # center is already given in camera coord system.
    box.translate(center + np.array([0, -wlh[2] / 2, 0]))

    # 3: Transform to KITTI LIDAR coord system. First transform from rectified
    # camera to camera, then
    # camera to KITTI lidar.
    box.rotate(Quaternion(matrix=transforms.r0_rect).inverse)
    box.translate(-transforms.velo_to_cam_translation)
    box.rotate(Quaternion(matrix=transforms.velo_to_cam_rotation).inverse)

    # Set score or NaN.
    box.score = score

    # Set dummy velocity.
    box.velocity = np.array((0.0, 0.0, 0.0))
    box = BBox3d(
        translation=box.center,
        size=box.wlh,
        rotation=box.orientation,
        label=box.name,
        score=box.score,
        velocity=box.velocity,
        sample_token=bbox.sample_idx,
    )
    return box


def read_kitti_calib(filename):
    """Read the camera 2 calibration matrix from box text file"""

    with open(filename) as f:
        for line in f:
            data = line.split(" ")
            if data[0] == "P2:":
                calib = np.array([float(x) for x in data[1:13]])
                return calib.reshape((3, 4))

    raise FileNotFoundError(
        "Could not find entry for P2 in calib file {}".format(filename)
    )


def read_kitti_objects(filename):
    objects = list()
    with open(filename, "r") as fp:

        # Each line represents box single object
        for line in fp:
            objdata = line.split(" ")
            if not (14 <= len(objdata) <= 15):
                raise IOError("Invalid KITTI object file {}".format(filename))

            # Parse object data
            objects.append(
                KittiBox3d(
                    label=objdata[0],
                    dimensions=[
                        float(objdata[10]),
                        float(objdata[8]),
                        float(objdata[9]),
                    ],
                    position=[float(p) for p in objdata[11:14]],
                    angle=float(objdata[14]),
                    score=float(objdata[15]) if len(objdata) == 16 else 1.0,
                    sample_idx=os.path.basename(filename),
                )
            )
    return objects


class Kitti(Dataset):
    """
    dataloader for kitti dataset
    http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d
    """

    def __init__(
        self,
        root=os.path.join(const.DEFAULT_DATA_ROOT, "kitti"),
        split="train",
        indices_file: str = None,
    ):
        """

        Args:
            root: path to where data already exists or where it will be
            downloaded to
            split: which split of the data to use. Can be: 'train', 'test',
            'val', 'trainval'. Can either specify split
            or indices file but not both.
            indices_file: file containing indices to use. Can either specify
            split or indices file but not both.
        """
        if split not in SPLITS:
            raise ValueError(
                f"invlaid value for split: {split},"
                f" possible values are: {SPLITS}"
            )
        if split is None and indices_file is None:
            raise ValueError(
                f"Cannot specify both indices file and split, must choose "
                f"one."
            )
        # todo should probably have separate val set with labels
        kitti_split = "testing" if split == "test" else "training"
        # self.root = os.path.join(root, kitti_split)
        self.root = root
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        self.split = split
        downloaded = self._check_downloaded()
        if not downloaded:
            logger.info(f"no local copy of dataset found.")
            self.download(cloud_path=KITTI_GCS_PATH)
        else:
            logger.info("local copy of dataset found, will not download")
        indices_file = indices_file or os.path.join(
            self.root, SAMPLEX_INDICES_FILE
        )
        self.indices = self._read_indices_file(filename=indices_file)
        self.root = os.path.join(root, kitti_split)

    def _check_downloaded(self):
        for z in ZIP_FILES:
            p = os.path.join(self.root, z)
            if not os.path.exists(p):
                logger.info(f"could not find file {p}")
                return False
        return True

    def __len__(self):
        return len(self.indices)

    def _get_calib_filename(self, idx):
        calib_file = os.path.join(self.root, f"calib/{idx:06d}.txt")
        return calib_file

    def _get_label_filename(self, idx):
        label_file = os.path.join(self.root, f"label_2/{idx:06d}.txt")
        return label_file

    def __getitem__(self, index):
        idx = self.indices[index]

        # Load image
        img_file = os.path.join(self.root, f"image_2/{idx:06d}.png")
        image = Image.open(img_file)

        # Load calibration matrix
        calib_file = os.path.join(self.root, f"calib/{idx:06d}.txt")
        calib = read_kitti_calib(calib_file)
        nu_transform = KittiTransforms(calib_filename=calib_file)

        # Load annotations
        label_file = os.path.join(self.root, f"label_2/{idx:06d}.txt")
        objects = read_kitti_objects(label_file)
        bboxes = [
            convert_kitti2nu(bbox=o, transforms=nu_transform) for o in objects
        ]
        return idx, image, calib, bboxes

    def _read_indices_file(self, filename):
        """

        Args:
            filename: path to file which contains kitti sample indices_file

        Returns: list of indices_file

        """
        with open(filename) as f:
            return [int(val) for val in f]

    def _download_sample_indices_file(
        self, *, cloud_client, object_key=None, local_file=None
    ):
        local_file = local_file or os.path.join(self.root, SAMPLEX_INDICES_FILE)
        object_key = object_key or f"{KITTI_GCS_PATH}/splits/{self.split}.txt"
        cloud_client.download(
            local_path=self.root, bucket=const.GCS_BUCKET, key=object_key
        )
        return local_file

    def download_kitti_zips(self, cloud_client, cloud_path=KITTI_GCS_PATH):
        calib_zip_key = f"{cloud_path}/data_object_calib.zip"
        left_images_zip_key = f"{cloud_path}/data_object_image_2.zip"
        left_image_labels_zip_key = f"{cloud_path}/data_object_label_2.zip"
        all_zips = [
            calib_zip_key,
            left_images_zip_key,
            left_image_labels_zip_key,
        ]
        local_zips = []
        for z in all_zips:
            local_path = os.path.join(self.root, z.split("/")[-1])
            cloud_client.download(
                local_path=self.root, bucket=const.GCS_BUCKET, key=z
            )
            local_zips.append(local_path)

        calib_zip, local_left_images_zip, local_labels_zip = [
            os.path.abspath(z) for z in local_zips
        ]
        return calib_zip, local_left_images_zip, local_labels_zip

    def _unzip2dir(self, *, zip_path, src, dst):
        """

        Args:
            zip_path: path to zip file
            src: the path within the unziped files to the file (or dir) to move
            dst: where to move the file (or dir) specified in src to


        """
        logger.info(f"extracting from {src} to {dst} ")
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(zip_path, "r") as zip_dir:
                zip_dir.extractall(tmp)
                shutil.move(os.path.join(tmp, src), dst)

    def download(self, cloud_path=KITTI_GCS_PATH):
        logger.info(f"downloading kitti dataset from cloud storage")
        # todo is currently only downloading left color images
        cloud_client = GCSClient()
        self._download_sample_indices_file(cloud_client=cloud_client)
        calib_zip, left_images_zip, labels_zip = self.download_kitti_zips(
            cloud_client=cloud_client
        )
        with zipfile.ZipFile(left_images_zip, "r") as zip_ref:
            zip_ref.extractall(self.root)
        testing_dir = os.path.join(self.root, "testing")
        training_dir = os.path.join(self.root, "training")
        self._unzip2dir(
            zip_path=calib_zip,
            src=os.path.join("testing", "calib"),
            dst=os.path.join(testing_dir, "calib"),
        )
        self._unzip2dir(
            zip_path=calib_zip,
            src=os.path.join("training", "calib"),
            dst=os.path.join(training_dir, "calib"),
        )
        self._unzip2dir(
            zip_path=labels_zip,
            src=os.path.join("training", "label_2"),
            dst=os.path.join(training_dir, "label_2"),
        )
