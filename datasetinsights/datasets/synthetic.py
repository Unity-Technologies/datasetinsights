""" Simulation Dataset Catalog
"""

import fcntl
import glob
import logging
import os
import shutil
from pathlib import Path

from PIL import Image
from pyquaternion import Quaternion

from datasetinsights.datasets.unity_perception import (
    AnnotationDefinitions,
    Captures,
)
from datasetinsights.datasets.unity_perception.tables import SCHEMA_VERSION
from datasetinsights.io.bbox import BBox2D, BBox3D

from .exceptions import DatasetNotFoundError

logger = logging.getLogger(__name__)


def read_bounding_box_3d(annotation, label_mappings=None):
    """ Convert dictionary representations of 3d bounding boxes into objects
    of the BBox3d class

    Args:
        annotation (List[dict]): 3D bounding box annotation
        label_mappings (dict): a dict of {label_id: label_name} mapping

    Returns:
        A list of 3d bounding box objects
    """

    bboxes = []

    for b in annotation:
        label_id = b["label_id"]
        translation = b["translation"]
        size = b["size"]
        rotation = b["rotation"]
        rotation = Quaternion(
            b=rotation[0], c=rotation[1], d=rotation[2], a=rotation[3]
        )

        if label_mappings and label_id not in label_mappings:
            continue
        box = BBox3D(
            translation=translation,
            size=size,
            label=label_id,
            sample_token=0,
            score=1,
            rotation=rotation,
        )
        bboxes.append(box)

    return bboxes


def read_bounding_box_2d(annotation, label_mappings=None):
    """Convert dictionary representations of 2d bounding boxes into objects
    of the BBox2D class

    Args:
        annotation (List[dict]): 2D bounding box annotation
        label_mappings (dict): a dict of {label_id: label_name} mapping

    Returns:
        A list of 2D bounding box objects
    """
    bboxes = []
    for b in annotation:
        label_id = b["label_id"]
        x = b["x"]
        y = b["y"]
        w = b["width"]
        h = b["height"]
        if label_mappings and label_id not in label_mappings:
            continue
        box = BBox2D(label=label_id, x=x, y=y, w=w, h=h)
        bboxes.append(box)

    return bboxes


class SynDetection2D:
    """Synthetic dataset for 2D object detection.

    During the class instantiation, it would check whether the data files
    such as annotations.json, images.png are present, if not it'll check
    whether a compressed dataset file is present which contains the necessary
    files, if not it'll raise an error.

    See synthetic dataset schema documentation for more details.
    <https://datasetinsights.readthedocs.io/en/latest/Synthetic_Dataset_Schema.html>

    Attributes:
        catalog (list): catalog of all captures in this dataset
        transforms: callable transformation that applies to a pair of
            capture, annotation. Capture is the information captured by the
            sensor, in this case an image, and annotations, which in this
            dataset are 2d bounding box coordinates and labels.
        label_mappings (dict): a dict of {label_id: label_name} mapping
    """

    ARCHIVE_FILE = "SynthDet.zip"
    SUBFOLDER = "synthetic"

    def __init__(
        self,
        *,
        data_path=None,
        transforms=None,
        version=SCHEMA_VERSION,
        def_id=4,
        **kwargs,
    ):
        """
        Args:
            data_path (str): Directory of the dataset
            transforms: callable transformation that applies to a pair of
            capture, annotation.
            version(str): synthetic dataset schema version
            def_id (int): annotation definition id used to filter results
        """
        self._data_path = self._preprocess_dataset(data_path)

        captures = Captures(self._data_path, version)
        annotation_definition = AnnotationDefinitions(self._data_path, version)
        catalog = captures.filter(def_id)
        self.catalog = self._cleanup(catalog)
        init_definition = annotation_definition.get_definition(def_id)
        self.label_mappings = {
            m["label_id"]: m["label_name"] for m in init_definition["spec"]
        }

        self.transforms = transforms

    def __getitem__(self, index):
        """
        Get the image and corresponding bounding boxes for that index
        Args:
            index:

        Returns (Tuple(Image,List(BBox2D))): Tuple comprising the image and
        bounding boxes found in that image with transforms applied.

        """
        cap = self.catalog.iloc[index]
        capture_file = cap.filename
        ann = cap["annotation.values"]

        capture = Image.open(os.path.join(self._data_path, capture_file))
        capture = capture.convert("RGB")  # Remove alpha channel
        annotation = read_bounding_box_2d(ann, self.label_mappings)

        if self.transforms:
            capture, annotation = self.transforms(capture, annotation)

        return capture, annotation

    def __len__(self):
        return len(self.catalog)

    def _cleanup(self, catalog):
        """
        remove rows with captures that having missing files and remove examples
        which have no annotations i.e. an image without any objects
        Args:
            catalog (pandas dataframe):

        Returns: dataframe without rows corresponding to captures that have
        missing files and removes examples which have no annotations i.e. an
        image without any objects.

        """
        catalog = self._remove_captures_with_missing_files(
            self._data_path, catalog
        )
        catalog = self._remove_captures_without_bboxes(catalog)

        return catalog

    @staticmethod
    def _remove_captures_without_bboxes(catalog):
        """Remove captures without bounding boxes from catalog

        Args:
            catalog (pd.Dataframe): The loaded catalog of the dataset

        Returns:
            A pandas dataframe with empty bounding boxes removed
        """
        keep_mask = catalog["annotation.values"].apply(len) > 0

        return catalog[keep_mask]

    @staticmethod
    def _remove_captures_with_missing_files(root, catalog):
        """Remove captures where image files are missing

        During the synthetic dataset download process, some of the files might
        be missing due to temporary http request issues or url corruption.
        We should remove these captures from catalog so that it does not
        stop the training pipeline.

        Args:
            catalog (pd.Dataframe): The loaded catalog of the dataset

        Returns:
            A pandas dataframe of the catalog with missing files removed
        """

        def exists(capture_file):
            path = Path(root) / capture_file

            return path.exists()

        keep_mask = catalog.filename.apply(exists)

        return catalog[keep_mask]

    @staticmethod
    def _preprocess_dataset(data_path):
        """ Preprocess dataset inside data_path and un-archive if necessary.

        Args:
            data_path (str): Path where dataset is stored.

        Return:
            Path of the dataset files.
        """
        archive_file = Path(data_path) / SynDetection2D.ARCHIVE_FILE
        if archive_file.exists():
            file_descriptor = os.open(archive_file, os.O_RDONLY)

            try:
                fcntl.flock(file_descriptor, fcntl.LOCK_EX)

                unarchived_path = Path(data_path) / SynDetection2D.SUBFOLDER
                if not SynDetection2D.is_dataset_files_present(unarchived_path):
                    shutil.unpack_archive(
                        filename=archive_file, extract_dir=unarchived_path
                    )

                return unarchived_path
            finally:
                os.close(file_descriptor)
        elif SynDetection2D.is_dataset_files_present(data_path):
            # This is for dataset generated by unity simulation.
            # In this case, all data are downloaded directly in the data_path
            return data_path
        else:

            raise DatasetNotFoundError(
                f"Expecting a file {archive_file} under {data_path} or files "
                f"directly exist under {data_path}"
            )

    @staticmethod
    def is_dataset_files_present(data_path):
        return os.path.isdir(data_path) and any(glob.glob(f"{data_path}/**/*"))
