""" Simulation Dataset Catalog
"""
import logging
import os
import zipfile
from collections import namedtuple
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split

import datasetinsights.constants as const
from datasetinsights.datasets.unity_perception import (
    AnnotationDefinitions,
    Captures,
)
from datasetinsights.datasets.unity_perception.tables import SCHEMA_VERSION
from datasetinsights.io.bbox import BBox2D
from datasetinsights.io.download import download_file, validate_checksum
from datasetinsights.io.exceptions import ChecksumError

from .base import Dataset

logger = logging.getLogger(__name__)
PUBLIC_SYNTHETIC_PATH = (
    "https://storage.googleapis.com/datasetinsights/data/synthetic"
)
SyntheticTable = namedtuple(
    "SyntheticTable", ("version", "filename", "source_uri", "checksum")
)

DEFAULT_TRAIN_SPLIT_RATIO = 0.9
TRAIN = "train"
VAL = "val"
ALL = "all"
VALID_SPLITS = (TRAIN, VAL, ALL)


def _get_split(*, split, catalog, train_percentage=0.9, random_seed=47):
    """

    Args:
        split (str): can be 'train', 'val' or 'all'
        catalog (pandas Dataframe): dataframe which will be divided into splits
        train_percentage (float): percentage of dataframe to put in train split
        random_seed (int): random seed used for splitting dataset into train
        and val

    Returns: catalog (dataframe) divided into correct split

    """
    if split == ALL:
        logger.info(f"spit specified was 'all' using entire synthetic dataset")
        return catalog
    train, val = train_test_split(
        catalog, train_size=train_percentage, random_state=random_seed
    )
    if split == TRAIN:
        catalog = train
        catalog.index = [i for i in range(len(catalog))]
        logger.info(
            f"split specified was {TRAIN}, using "
            f"{train_percentage*100:.2f}% of the dataset"
        )
        return catalog
    elif split == VAL:
        catalog = val
        catalog.index = [i for i in range(len(catalog))]
        logger.info(
            f"split specified was {VAL} using "
            f"{(1-train_percentage)*100:.2f}% of the dataset "
        )
        return catalog
    else:
        raise ValueError(
            f"split provided was {split} but only valid "
            f"splits are: {VALID_SPLITS}"
        )


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


class SynDetection2D(Dataset):
    """Synthetic dataset for 2D object detection.

    During the class instantiation, it would check whehter the data is
    downloaded or not or if USIM run-execution-id or manifest file is
    provided. If there is no dataset and USIM run-execution-id or manifest file
    it would raise an error.

    Current public synthdet version are as follows:
        v1: Subset SynthDet dataset containing around 5k images.

    Attributes:
        root (str): root directory of the dataset
        catalog (list): catalog of all captures in this dataset
        transforms: callable transformation that applies to a pair of
            capture, annotation. Capture is the information captured by the
            sensor, in this case an image, and annotations, which in this
            dataset are 2d bounding box coordinates and labels.
        split (str): indicate split type of the dataset (train|val|test)
        label_mappings (dict): a dict of {label_id: label_name} mapping
    """

    SYNTHETIC_DATASET_TABLES = {
        "v1": SyntheticTable(
            "v1",
            "SynthDet.zip",
            f"{PUBLIC_SYNTHETIC_PATH}/SynthDet.zip",
            390326588,
        ),
    }

    def __init__(
        self,
        *,
        data_root=const.DEFAULT_DATA_ROOT,
        split="all",
        transforms=None,
        version=SCHEMA_VERSION,
        def_id=4,
        train_split_ratio=DEFAULT_TRAIN_SPLIT_RATIO,
        random_seed=47,
        **kwargs,
    ):
        """
        Args:
            data_root (str): root directory prefix of dataset
            transforms: callable transformation that applies to a pair of
            capture, annotation.
            version(str): synthetic dataset schema version
            def_id (int): annotation definition id used to filter results
            random_seed (int): random seed used for splitting dataset into
                train and val
        """
        self.dataset_directory = os.path.join(data_root, const.SYNTHETIC_SUBFOLDER)

        captures = Captures(self.dataset_directory, version)
        annotation_definition = AnnotationDefinitions(
            self.dataset_directory, version
        )
        catalog = captures.filter(def_id)
        self.catalog = self._cleanup(catalog)
        init_definition = annotation_definition.get_definition(def_id)
        self.label_mappings = {
            m["label_id"]: m["label_name"] for m in init_definition["spec"]
        }

        if split not in VALID_SPLITS:
            raise ValueError(
                f"split provided was {split} but only valid "
                f"splits are: {VALID_SPLITS}"
            )
        self.split = split
        self.transforms = transforms
        self.catalog = _get_split(
            split=split,
            catalog=self.catalog,
            train_percentage=train_split_ratio,
            random_seed=random_seed,
        )

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

        capture = Image.open(os.path.join(self.dataset_directory, capture_file))
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
            self.dataset_directory, catalog
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
    def download(data_root, version):
        """Downloads dataset zip file and unzips it.

        Args:
            data_root (str): Path where to download the dataset.
            version (str): version of GroceriesReal dataset, e.g. "v1"

        Raises:
             ValueError if the dataset version is not supported
             ChecksumError if the download file checksum does not match
             DownloadError if the download file failed

        Note: Synthetic dataset is downloaded and unzipped to
        data_root/synthetic.
        """
        if version not in SynDetection2D.SYNTHETIC_DATASET_TABLES.keys():
            raise ValueError(
                f"A valid dataset version is required. Available versions are:"
                f"{SynDetection2D.SYNTHETIC_DATASET_TABLES.keys()}"
            )

        source_uri = SynDetection2D.SYNTHETIC_DATASET_TABLES[version].source_uri
        expected_checksum = SynDetection2D.SYNTHETIC_DATASET_TABLES[
            version
        ].checksum
        dataset_file = SynDetection2D.SYNTHETIC_DATASET_TABLES[version].filename

        extract_folder = os.path.join(data_root, const.SYNTHETIC_SUBFOLDER)
        dataset_path = os.path.join(extract_folder, dataset_file)

        if os.path.exists(dataset_path):
            logger.info("The dataset file exists. Skip download.")
            try:
                validate_checksum(dataset_path, expected_checksum)
            except ChecksumError:
                logger.info(
                    "The checksum of the previous dataset mismatches. "
                    "Delete the previously downloaded dataset."
                )
                os.remove(dataset_path)

        if not os.path.exists(dataset_path):
            logger.info(f"Downloading dataset to {extract_folder}.")
            download_file(source_uri, dataset_path)
            try:
                validate_checksum(dataset_path, expected_checksum)
            except ChecksumError as e:
                logger.info("Checksum mismatch. Delete the downloaded files.")
                os.remove(dataset_path)
                raise e

        SynDetection2D.unzip_file(
            filepath=dataset_path, destination=extract_folder
        )

    @staticmethod
    def unzip_file(filepath, destination):
        """Unzips a zip file to the destination and delete the zip file.

        Args:
            filepath (str): File path of the zip file.
            destination (str): Path where to unzip contents of zipped file.
        """
        with zipfile.ZipFile(filepath) as file:
            logger.info(f"Unzipping file from {filepath} to {destination}")
            file.extractall(destination)
        os.remove(filepath)
