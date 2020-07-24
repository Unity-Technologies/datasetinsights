""" Simulation Dataset captures
"""
import logging
import os
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split

import datasetinsights.constants as const
from datasetinsights.data.bbox import BBox2D
from datasetinsights.data.simulation import AnnotationDefinitions, Captures
from datasetinsights.data.simulation.tables import SCHEMA_VERSION

from .base import Dataset

logger = logging.getLogger(__name__)
DEFAULT_TRAIN_SPLIT_RATIO = 0.9
TRAIN = "train"
VAL = "val"
ALL = "all"
VALID_SPLITS = (TRAIN, VAL, ALL)


def _get_split(*, split, captures, train_percentage=0.9, random_seed=47):
    """
    Args:
        split (str): can be 'train', 'val' or 'all'
        captures (Dask.Dataframe): dataframe which will be divided into splits
        train_percentage (float): percentage of dataframe to put in train split
        random_seed (int): random seed used for splitting dataset into train
        and val

    Returns: captures (Dask.Dataframe) divided into correct split

    TODO (YC) Should move train/val/test split as as seaprate step right after
    synthetic dataset is download. The Dataset object should not have to handle
    train/test splits here.
    """
    if split == ALL:
        logger.info(f"spit specified was 'all' using entire synthetic dataset")
        return captures
    train, val = train_test_split(
        captures, train_size=train_percentage, random_state=random_seed
    )
    if split == TRAIN:
        captures = train
        captures.index = [i for i in range(len(captures))]
        logger.info(
            f"split specified was {TRAIN}, using "
            f"{train_percentage*100:.2f}% of the dataset"
        )
        return captures
    elif split == VAL:
        captures = val
        captures.index = [i for i in range(len(captures))]
        logger.info(
            f"split specified was {VAL} using "
            f"{(1-train_percentage)*100:.2f}% of the dataset "
        )
        return captures
    else:
        raise ValueError(
            f"split provided was {split} but only valid "
            f"splits are: {VALID_SPLITS}"
        )


def read_bounding_box_2d(annotations, label_mappings=None):
    """Read 2d bounding boxes into list of BBox2D objects

    This method reads a table of 2D bounding box annotations store in
    a Dask.DataFrame with the same capture.id. Each row represents a single
    bounding box annotations. If the label_id of the given bounding box
    are not defined in label_mappings, this bounding box will be ignored.

    Args:
        annotations (Dask.DataFrame): 2D bounding box annotations store in
            DataFrame.
        label_mappings (dict): A dict of {label_id: label_name} mappings

    Returns:
        A list of BBox2D objects.
    """
    bboxes = []
    for _, box in annotations.iterrows():
        label_id = box[f"{Captures.VALUES_COLUMN}.label_id"]
        x = box[f"{Captures.VALUES_COLUMN}.x"]
        y = box[f"{Captures.VALUES_COLUMN}.y"]
        w = box[f"{Captures.VALUES_COLUMN}.width"]
        h = box[f"{Captures.VALUES_COLUMN}.height"]
        if label_mappings and label_id not in label_mappings:
            continue
        box = BBox2D(label=label_id, x=x, y=y, w=w, h=h)
        bboxes.append(box)

    return bboxes


class SynDetection2D(Dataset):
    """Synthetic dataset for 2D object detection.

    Attributes:
        root (str): root directory of the dataset
        captures (Dask.DataFrame): captures after filter in dataset
        annotations (Dask.DataFrame): annotations after filter by definition id
        transforms: callable transformation that applies to a pair of
            capture, annotation. Capture is the information captured by the
            sensor, in this case an image, and annotations, which in this
            dataset are 2d bounding box coordinates and labels.
        split (str): indicate split type of the dataset (train|val|test)
        label_mappings (dict): a dict of {label_id: label_name} mapping
    """

    def __init__(
        self,
        *,
        data_root=const.DEFAULT_DATA_ROOT,
        split="all",
        transforms=None,
        version=SCHEMA_VERSION,
        run_execution_id=None,
        def_id=4,
        train_split_ratio=DEFAULT_TRAIN_SPLIT_RATIO,
        random_seed=47,
        **kwargs,
    ):
        """ Initialize SynDetection2D

        Args:
            data_root (str): root directory prefix of dataset
            transforms: callable transformation that applies to a pair of
                capture, annotation.
            version (str): synthetic dataset schema version
            run_execution_id (str): USim run execution id, if this argument
                is provided then the class will attempt to download the data
                from USim. If the data has already been downloaded locally,
                then this argument should be None and the caller should pass
                in the location of the manifest_file for the manifest arg.
                For more information on Unity Simulations please see
                https://github.com/Unity-Technologies/Unity-Simulation-Docs
            def_id (int): annotation definition id used to filter results
            random_seed (int): random seed used for splitting dataset into
                train and val
        """
        self.root = os.path.join(
            data_root, const.SYNTHETIC_SUBFOLDER, run_execution_id
        )
        logger.info("Root directory of synthetic data: {self.root}")
        self.label_mappings = self._load_label_mappings(version, def_id)

        captures_table = Captures(self.root, version)
        captures = captures_table.filter_captures()
        captures = self._remove_captures_with_missing_files(
            self.root, captures
        )
        self.annotations = captures_table.filter_annotations(def_id)
        self.captures = self._remove_captures_without_bboxes(
            captures, self.annotations
        )

        if split not in VALID_SPLITS:
            raise ValueError(
                f"split provided was {split} but only valid "
                f"splits are: {VALID_SPLITS}"
            )
        self.split = split
        self.transforms = transforms
        self.captures = _get_split(
            split=split,
            captures=self.captures,
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
        cap = self.captures.iloc[index]
        capture_file = os.path.join(self.root, cap.filename)
        capture_id = cap.id
        capture = Image.open(capture_file)
        capture = capture.convert("RGB")  # Remove alpha channel

        ann = self.annotations.loc[capture_id]
        annotations = read_bounding_box_2d(ann, self.label_mappings)

        if self.transforms:
            capture, annotations = self.transforms(capture, annotations)

        return capture, annotations

    def __len__(self):
        return len(self.captures)

    @staticmethod
    def _remove_captures_with_missing_files(root, captures):
        """Remove captures where image files are missing

        During the synthetic dataset download process, some of the files might
        be missing due to temporary http request issues or url corruption.
        We should remove these captures from captures so that it does not
        stop the training pipeline.

        Args:
            captures (Dask.Dataframe): The loaded captures of the dataset

        Returns:
            A pandas dataframe of the captures with missing files removed
        """

        def exists(capture_file):
            path = Path(root) / capture_file

            return path.exists()

        keep_mask = captures.filename.apply(exists)

        return captures[keep_mask]

    @staticmethod
    def _remove_captures_without_bboxes(captures, annotations):
        """Remove captures without bounding boxes from captures

        Args:
            captures (Dask.Dataframe): The loaded captures of the dataset

        Returns:
            A pandas dataframe with empty bounding boxes removed
        """
        capture_ids = captures.id
        capture_ids_with_annotations = set(annotations.index)
        keep_mask = capture_ids.apply(
            lambda x: x in capture_ids_with_annotations
        )

        return captures[keep_mask]

    def _load_label_mappings(self, version, def_id):
        """Load label mappings.

        Args:
            version (str): synthetic dataset schema version
            def_id (str):

        Returns:
            dict: A dict containing {label_id: label_name} mappings.
        """
        annotation_definition = AnnotationDefinitions(self.root, version)

        filtered_def = annotation_definition.get_definition(def_id)
        label_mappings = {
            m["label_id"]: m["label_name"] for m in filtered_def["spec"]
        }

        return label_mappings
