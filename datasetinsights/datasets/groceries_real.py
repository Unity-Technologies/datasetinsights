import json
import logging
import os
import zipfile
from collections import namedtuple
from pathlib import Path

import tensorflow as tf
from google.protobuf import text_format
from PIL import Image

import datasetinsights.constants as const
from datasetinsights.io.bbox import BBox2D
from datasetinsights.io.download import download_file, validate_checksum
from datasetinsights.io.exceptions import ChecksumError, DownloadError
from datasetinsights.io.gcs import download_file_from_gcs

from .base import Dataset
from .exceptions import DatasetNotFoundError
from .protos import string_int_label_map_pb2

logger = logging.getLogger(__name__)
PUBLIC_GROCERIES_REAL_PATH = (
    "https://storage.googleapis.com/datasetinsights/data/groceries"
)
GroceriesRealTable = namedtuple(
    "GroceriesRealTable", ("version", "filename", "source_uri", "checksum")
)


class GroceriesReal(Dataset):
    """Unity's Groceries Real Dataset.

    During the class instantiation, it would check whehter the data is
    downloaded or not. If there is no dataset, it would raise an error.
    Please make sure you download the dataset before use this class.

    The dataset contain the following structures:

    * images/

        * IMG_4185.JPG
        * IMG_4159.JPG
        * ...
    * `annotation_definition.json` file stores label_id to label_name mappings
    * `annotations.json` file stores 2d bounding box records:

    .. code-block:: json
        [
            {
                "file_name": "IMG_4185.JPG",
                "annotations":[
                    {"label_id": 11, "bboxs":[1692, 1386, 768, 133}
                ]
            },
            ...
        ]
    * `groceries_real_<split>.txt` file which stores space deplimted list
    of the indices in the annotations.json file, for a given data split.


    Attributes:
        root (str): Local directory where groceries_real dataset is saved
        split (str): Indicate split type of the dataset. Allowed value are
            (train|val|test|test_high_ratio|test_low_ratio)
            test_high_ratio: Split of test dataset with high
            foreground-to-background ratio.
            test_low_ratio: Split of test dataset with low
            foreground-to-background ratio.
        transforms: Callable transformation
        annotations (list): A list of dict that contain image file path
            and 2D bounding box annotations.
        split_indices (list): A list of indices for this dataset split.
        label_mappings (dict): a dict of {label_id: label_name} mapping
        version (str): version of GroceriesReal dataset, e.g. "v3".
        default version="v3".
    """

    LOCAL_PATH = "groceries"
    SPLITS = {
        "train": "groceries_real_train.txt",
        "val": "groceries_real_val.txt",
        "test": "groceries_real_test.txt",
        # Split train dataset into small/medium (10%/50%) subset
        "train_small": "groceries_real_train_small.txt",
        "train_medium": "groceries_real_train_medium.txt",
        # Split test dataset into high/low foreground to background ratio
        "test_high_ratio": "groceries_real_test_high_ratio.txt",
        "test_low_ratio": "groceries_real_test_low_ratio.txt",
        # Split test datset into high/low contrast. Images in high contrast
        # split tend to have more complicated shadow pattern
        "test_low_contrast": "groceries_real_test_low_contrast.txt",
        "test_high_contrast": "groceries_real_test_high_contrast.txt",
    }

    GROCERIES_REAL_DATASET_TABLES = {
        "v3": GroceriesRealTable(
            "v3", "v3.zip", f"{PUBLIC_GROCERIES_REAL_PATH}/v3.zip", 2322380006,
        ),
    }

    ANNOTATION_FILE = "annotations.json"
    DEFINITION_FILE = "annotation_definitions.json"

    def __init__(
        self,
        *,
        data_root=const.DEFAULT_DATA_ROOT,
        split="train",
        transforms=None,
        version="v3",
        **kwargs,
    ):
        """
        Args:
            data_root (str): Root directory prefix of datasets
            split (str): Indicate split type of the dataset.
            transforms: callable transformation
            version (str): version of GroceriesReal dataset
        """
        valid_splits = tuple(self.SPLITS.keys())
        if split not in valid_splits:
            raise ValueError(
                f"Invalid value for split: {split}. Allowed values "
                f"are: {valid_splits}"
            )
        self.split = split

        valid_versions = tuple(self.GROCERIES_REAL_DATASET_TABLES.keys())
        if version not in valid_versions:
            raise ValueError(
                f"A valid dataset version should be set. "
                f"Available versions are: {valid_versions}"
            )
        logger.info(
            f"Using split {split} and version {version} for groceries real "
            "dataset."
        )

        self.version = version
        self.root = os.path.join(data_root, self.LOCAL_PATH)
        self.transforms = transforms
        if not os.path.isdir(os.path.join(self.root, f"{version}")):
            raise DatasetNotFoundError(
                "Cannot find the dataset. Please download it first."
            )
        self.annotations = self._load_annotations()
        self.split_indices = self._load_split_indices()
        self.label_mappings = self._load_label_mappings()

    def __getitem__(self, idx):
        """
        Args:
            idx (int): index of the dataset catalog

        Returns:
            tuple: A pair of PIL image and list of BBox2D object.
        """
        data_idx = self.split_indices[idx]
        image_data = self.annotations[data_idx]
        image_filename = image_data["file_name"]
        annotations = image_data["annotations"]
        image_filename = self._filepath(os.path.join("images", image_filename))
        image = Image.open(image_filename)
        bboxes = [self._convert_to_bbox2d(ann) for ann in annotations]
        if self.transforms:
            image, bboxes = self.transforms(image, bboxes)

        return image, bboxes

    def __len__(self):
        return len(self.split_indices)

    def _filepath(self, filename):
        """Local file path relative to root
        """
        return os.path.join(self.root, self.version, filename)

    @staticmethod
    def _download_http(source_uri, dest_path, version):
        """ Download dataset from Public HTTP URL.

        Args:
            source_uri (str): source url where the file should be downloaded
            dest_path (str): destination path of the file

        Raises:
            DownloadError if the download file failed
            ChecksumError if the download file checksum does not match
        """

        try:
            logger.info("Downloading the dataset.")
            download_file(source_uri=source_uri, dest_path=dest_path)
        except DownloadError as e:
            logger.info(
                f"The request download from {source_uri} -> {dest_path} can't "
                f"be completed."
            )
            raise e
        expected_checksum = GroceriesReal.GROCERIES_REAL_DATASET_TABLES[
            version
        ].checksum
        try:
            validate_checksum(dest_path, expected_checksum)
        except ChecksumError as e:
            logger.info("Checksum mismatch. Delete the downloaded files.")
            os.remove(dest_path)
            raise e

    @staticmethod
    def _extract_file(dest_path, root_dir):
        """ Unzip the downloaded file.
        """
        logger.info("Unzipping the dataset file.")
        with zipfile.ZipFile(dest_path, "r") as zip_dir:
            zip_dir.extractall(root_dir)

    @staticmethod
    def download(data_root, version):
        """ Download dataset from Public HTTP URL.

        If the file already exists and the checksum matches, it will skip the
        download step. If not, it would delete the previous file and download
        it again. If the file doesn't exist, it would download the file.

        Args:
            data_root (str): Root directory prefix of datasets
            version (str): version of GroceriesReal dataset, e.g. "v3"

        Raises:
            ValueError if the dataset version is not supported
            ChecksumError if the download file checksum does not match
            DownloadError if the download file failed
        """
        if version not in GroceriesReal.GROCERIES_REAL_DATASET_TABLES.keys():
            raise ValueError(
                f"A valid dataset version is required. Available versions are:"
                f"{GroceriesReal.GROCERIES_REAL_DATASET_TABLES.keys()}"
            )
        dest_path = os.path.join(
            data_root, GroceriesReal.LOCAL_PATH, f"{version}.zip"
        )
        expected_checksum = GroceriesReal.GROCERIES_REAL_DATASET_TABLES[
            version
        ].checksum
        extract_folder = os.path.join(data_root, GroceriesReal.LOCAL_PATH)
        if os.path.exists(dest_path):
            logger.info("The dataset file exists. Skip download.")
            try:
                validate_checksum(dest_path, expected_checksum)
            except ChecksumError:
                logger.info(
                    "The checksum of the previous dataset mismatches. "
                    "Delete the previously downloaded dataset."
                )
                os.remove(dest_path)
        if not os.path.exists(dest_path):
            source_uri = GroceriesReal.GROCERIES_REAL_DATASET_TABLES[
                version
            ].source_uri
            GroceriesReal._download_http(source_uri, dest_path, version)
        GroceriesReal._extract_file(dest_path, extract_folder)

    def _load_annotations(self):
        """Load annotation from annotations.json file

        Returns:
            list: A list of annotations stored in annotations.json file
        """
        annotation_file = self._filepath(self.ANNOTATION_FILE)
        with open(annotation_file) as f:
            json_data = json.load(f)

        return json_data

    def _load_split_indices(self):
        """Load the data indices txt file.

        Returns:
            list: A list of data indices in annotations.json file.
        """
        split_file = self.SPLITS.get(self.split)
        indices_file = self._filepath(split_file)

        with open(indices_file) as txt_file:
            idx_data = [int(i) for i in txt_file.readline().split()]

        return idx_data

    def _load_label_mappings(self):
        """Load label mappings.

        Returns:
            dict: A dict containing {label_id: label_name} mappings.
        """
        definition_file = self._filepath(self.DEFINITION_FILE)
        with open(definition_file) as json_file:
            init_definition = json.load(json_file)["annotation_definitions"][0]
            label_mappings = {
                m["label_id"]: m["label_name"] for m in init_definition["spec"]
            }

        return label_mappings

    @staticmethod
    def _convert_to_bbox2d(single_bbox):
        """Convert the bbox record to BBox2D objects.

        Args:
            single_bbox (dict): raw bounding box information

        Return:
            canonical_bbox (BBox2D): canonical bounding box
        """
        label = single_bbox["label_id"]
        bbox = single_bbox["bbox"]

        canonical_bbox = BBox2D(
            x=bbox[0], y=bbox[1], w=bbox[2], h=bbox[3], label=label
        )
        return canonical_bbox


class GoogleGroceriesReal(Dataset):
    """Google GroceriesReal Dataset

    This dataset include groceries images and annotations in this paper_.
    Please reach out to the authors to request the dataset.

    The dataset should contain the following structures:

    * Train Data/
        file0.tfexample
        file1.tfexample
        ...
    * Eval Data/
        file234.tfexample
        ...
    * Label Map/label_map_with_path_64_retail.txt

    .. _paper: https://arxiv.org/abs/1902.09967

    Attributes:
        root (str): Local directory of the dataset is.
        split (str): Indicate split type of the dataset.
            Allowed value are (train|test).
        transforms: Callable transformation
        label_mappings (dict): a dict of {label_id: label_name} mapping
            loaded from label_map_with_path_64_retail.txt
    """

    GCS_PATH = "data/google_groceries"
    LOCAL_PATH = "google_groceries"
    SPLITS_ZIP = {
        "train": "Train Data.zip",
        # Their dataset does not provide separate validateion/test dataset.
        # Using "eval" as test dataset.
        "test": "Eval Data.zip",
    }
    SPLITS_FOLDER = {
        "train": "Train Data",
        "test": "Eval Data",
    }
    LABEL_ZIP = "Label Map.zip"
    LABEL_FILE = "Label Map/label_map_with_path_64_retail.txt"
    TF_FEATURES = {
        "image/channels": tf.io.FixedLenFeature([], tf.int64),
        "image/encoded": tf.io.FixedLenFeature([], tf.string),
        "image/format": tf.io.FixedLenFeature([], tf.string),
        "image/width": tf.io.FixedLenFeature([], tf.int64),
        "image/height": tf.io.FixedLenFeature([], tf.int64),
        "image/object/bbox/occluded": tf.io.FixedLenFeature([], tf.float32),
        "image/object/bbox/xmax": tf.io.VarLenFeature(tf.float32),
        "image/object/bbox/xmin": tf.io.VarLenFeature(tf.float32),
        "image/object/bbox/ymax": tf.io.VarLenFeature(tf.float32),
        "image/object/bbox/ymin": tf.io.VarLenFeature(tf.float32),
        "image/object/class/label": tf.io.VarLenFeature(tf.int64),
        "image/object/class/text": tf.io.VarLenFeature(tf.string),
    }

    def __init__(self, data_root, split="train", transforms=None):
        """Initialize GoogleGroceriesReal Dataset

        Args:
            data_root (str): Root directory prefix of datasets.
            split (str): Indicate split type of the dataset.
            transforms: callable transformations.
        """
        valid_splits = tuple(self.SPLITS_ZIP.keys())
        if split not in valid_splits:
            raise ValueError(
                f"Invalid value for split: {split}. Allowed values "
                f"are: {valid_splits}"
            )
        self.split = split
        self.root = os.path.join(data_root, self.LOCAL_PATH)
        self.download()

        self._tfexample_files = self._glob_tfexamples()
        self.transforms = transforms
        self.label_mappings = self._load_label_mappings()

    def __getitem__(self, index):
        """
        Args:
            index (int): index of the dataset catalog

        Returns:
            tuple: A pair of PIL image and list of BBox2D objects.
        """
        tfexample_file = self._tfexample_files[index]
        with open(tfexample_file, "rb") as f:
            record = f.read()
        raw_record = tf.io.parse_single_example(record, self.TF_FEATURES)
        image = self._load_image(raw_record)
        bboxes = self._load_bounding_boxes(raw_record)

        if self.transforms:
            image, bboxes = self.transforms(image, bboxes)

        return image, bboxes

    def __len__(self):
        return len(self._tfexample_files)

    def download(self):
        """Download dataset from GCS
        """
        cloud_path = f"gs://{const.GCS_BUCKET}/{self.GCS_PATH}"
        # download label file
        label_zip = download_file_from_gcs(
            cloud_path, self.root, self.LABEL_ZIP
        )
        with zipfile.ZipFile(label_zip, "r") as zip_dir:
            zip_dir.extractall(self.root)

        # download tfexamples for a dataset split
        tfexamples_zip = download_file_from_gcs(
            cloud_path, self.root, self.SPLITS_ZIP.get(self.split)
        )
        with zipfile.ZipFile(tfexamples_zip, "r") as zip_dir:
            zip_dir.extractall(self.root)

    def _glob_tfexamples(self):
        split_folder = self.SPLITS_FOLDER.get(self.split)
        path = Path(self.root) / split_folder
        filenames = [str(p) for p in path.glob("*.tfexample")]

        return filenames

    def _load_image(self, raw_record):
        tf_img = raw_record["image/encoded"]
        np_img = tf.io.decode_image(tf_img).numpy()
        pil_img = Image.fromarray(np_img)

        return pil_img

    def _load_bounding_boxes(self, raw_record):
        img_width = raw_record["image/width"].numpy()
        img_height = raw_record["image/height"].numpy()
        label = tf.sparse.to_dense(
            raw_record["image/object/class/label"]
        ).numpy()
        xmin = (
            tf.sparse.to_dense(raw_record["image/object/bbox/xmin"]).numpy()
            * img_width
        )
        xmax = (
            tf.sparse.to_dense(raw_record["image/object/bbox/xmax"]).numpy()
            * img_width
        )
        ymin = (
            tf.sparse.to_dense(raw_record["image/object/bbox/ymin"]).numpy()
            * img_height
        )
        ymax = (
            tf.sparse.to_dense(raw_record["image/object/bbox/ymax"]).numpy()
            * img_height
        )

        width = xmax - xmin
        height = ymax - ymin
        bboxes = [
            BBox2D(label, x, y, w, h)
            for label, x, y, w, h in zip(label, xmin, ymin, width, height)
        ]

        return bboxes

    def _load_label_mappings(self):
        label_file = os.path.join(self.root, self.LABEL_FILE)
        with open(label_file, "rb") as f:
            label_map = text_format.Parse(
                f.read(), string_int_label_map_pb2.StringIntLabelMap()
            )

        label_mapppings = {item.id: item.name for item in label_map.item}

        return label_mapppings
