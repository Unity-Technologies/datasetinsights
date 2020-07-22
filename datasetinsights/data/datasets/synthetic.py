""" Simulation Dataset Catalog
"""
import logging
import os
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split

import datasetinsights.constants as const
from datasetinsights.data.bbox import BBox2D
from datasetinsights.data.simulation import AnnotationDefinitions, Captures
from datasetinsights.data.simulation.download import (
    Downloader,
    download_manifest,
)
from datasetinsights.data.simulation.tables import SCHEMA_VERSION

from .base import Dataset

logger = logging.getLogger(__name__)
SYNTHETIC_LOCAL_PATH = "synthetic"
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


def _download_captures(root, manifest_file):
    """Download captures for synthetic dataset
    Args:
        root (str): root directory where the dataset should be downloaded
        manifest_file (str): path to USim simulation manifest file
    """
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)

    dl = Downloader(manifest_file, root)
    dl.download_captures()
    dl.download_references()
    dl.download_binary_files()


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

    def __init__(
        self,
        *,
        data_root=const.DEFAULT_DATA_ROOT,
        split="all",
        transforms=None,
        manifest_file=None,
        run_execution_id=None,
        auth_token=None,
        version=SCHEMA_VERSION,
        def_id=4,
        train_split_ratio=DEFAULT_TRAIN_SPLIT_RATIO,
        random_seed=47,
        **kwargs,
    ):
        """
        Args:
            data_root (str): root directory prefix of dataset
            manifest_file (str): path to a manifest file. Use this argument
                if the synthetic data has already been downloaded. If the
                synthetic dataset hasn't been downloaded, leave this argument
                as None and provide the run_execution_id and auth_token and
                this class will download the dataset. For more information
                on Unity Simulations (USim) please see
                https://github.com/Unity-Technologies/Unity-Simulation-Docs
            transforms: callable transformation that applies to a pair of
            capture, annotation.
            version(str): synthetic dataset schema version
            def_id (int): annotation definition id used to filter results
            run_execution_id (str): USim run execution id, if this argument
                is provided then the class will attempt to download the data
                from USim. If the data has already been downloaded locally,
                then this argument should be None and the caller should pass
                in the location of the manifest_file for the manifest arg.
                For more information
                on Unity Simulations (USim) please see
                https://github.com/Unity-Technologies/Unity-Simulation-Docs
            auth_token (str): usim authorization token that can be used to
                interact with usim API to download manifest files. This token
                is necessary to download the dataset form USim. If the data is
                already stored locally, then this argument can be left as None.
                For more information
                on Unity Simulations (USim) please see
                https://github.com/Unity-Technologies/Unity-Simulation-Docs
            random_seed (int): random seed used for splitting dataset into
                train and val
        """
        if run_execution_id:
            manifest_file = os.path.join(
                data_root, SYNTHETIC_LOCAL_PATH, f"{run_execution_id}.csv"
            )
            download_manifest(
                run_execution_id,
                manifest_file,
                auth_token,
                project_id=const.DEFAULT_PROJECT_ID,
            )
        if manifest_file:
            subfolder = Path(manifest_file).stem
            self.root = os.path.join(data_root, SYNTHETIC_LOCAL_PATH, subfolder)
            self.download(manifest_file)
        else:
            logger.info(
                f"No manifest file is provided. Assuming the data root "
                f"directory {data_root} already contains synthetic dataset."
            )
            self.root = data_root

        captures = Captures(self.root, version)
        annotation_definition = AnnotationDefinitions(self.root, version)
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

        capture = Image.open(os.path.join(self.root, capture_file))
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
        catalog = self._remove_captures_with_missing_files(self.root, catalog)
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

    def download(self, manifest_file):
        """ Download captures of a given manifest file.

        Args:
            manifest_file (str): path to a manifest file
        """
        _download_captures(self.root, manifest_file)
