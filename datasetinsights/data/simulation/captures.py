""" Load Synthetic dataset captures and annotations tables
"""
import dask.bag as db
import pandas as pd

from datasetinsights.constants import DEFAULT_DATA_ROOT

from .exceptions import DefinitionIDError, NoCapturesError
from .tables import DATASET_TABLES, SCHEMA_VERSION, glob, load_table

from memory_profiler import profile


class Captures:
    """Load captures table

    A capture record stores the relationship between a captured file,
    a collection of annotations, and extra metadata that describes this
    capture. For more detail, see schema design here:

    :ref:`captures`

    Examples:
        >>> captures = Captures(data_root="/data")
        # captures class automatically loads the captures (e.g. lidar scan,
        image, depth map) and the annotations (e.g semantic segmentation
        labels, bounding boxes, etc.)
        >>> data = captures.filter(def_id="6716c783-1c0e-44ae-b1b5-7f068454b66e")
        # return the captures and annotations filtered by the annotation
        definition id

    Attributes:
        captures (pd.DataFrame): a collection of captures without annotations
        annotations (pd.DataFrame): a collection of annotations
    """  # noqa: E501 table command not be broken down into multiple lines

    TABLE_NAME = "captures"
    FILE_PATTERN = DATASET_TABLES[TABLE_NAME].file
    ANNOTATION_TABLE_INDEX = "capture.id"
    VALUES_COLUMN = "values"

    def __init__(self, data_root=DEFAULT_DATA_ROOT, version=SCHEMA_VERSION):
        """ Initialize Captures

        Args:
            data_root (str): the root directory of the dataset
            version (str): desired schema version
        """
        self.captures = self._load_captures(data_root, version)
        self.annotations = self._load_annotations(data_root, version)

    @profile
    def _load_captures(self, data_root, version):
        """Load captures except annotations.
        :ref:`captures`

        Args:
            data_root (str): the root directory of the dataset
            version (str): desired schema version

        Returns:
            A dask bag of captures except annotations
        """
        capture_files = glob(data_root, self.FILE_PATTERN)
        captures = db.from_sequence(capture_files)
        if captures.count().compute() == 0:
            raise NoCapturesError(f"Can't find captures files in {data_root}")

        def _remove_annotations(record):
            del record["annotations"]

            return record

        captures = (
            captures.map(
                lambda path: load_table(path, self.TABLE_NAME, version)
            )
            .flatten()
            .map(_remove_annotations)
        )

        return captures

    @profile
    def filter_captures(self, sensor_id=None, ego_id=None, modality=None):
        """Filter captures by specific sensor, ego or modality

        If None, no filter is applied
        """
        # TODO (YC) Implement filter here.
        return self.captures.to_dataframe().compute()

    @profile
    def _load_annotations(self, data_root, version):
        """Load annotations and capture IDs.
        :ref:`capture-annotation`

        Args:
            data_root (str): the root directory of the dataset
            version (str): desired schema version

        Returns:
            A Dask bag of annotations
        """
        capture_files = glob(data_root, self.FILE_PATTERN)
        captures = db.from_sequence(capture_files)

        def _annotation_record(capture):
            anns = []
            for ann in capture["annotations"]:
                ann[self.ANNOTATION_TABLE_INDEX] = capture["id"]
                anns.append(ann)

            return anns

        annotations = (
            captures.map(
                lambda path: load_table(path, self.TABLE_NAME, version)
            )
            .flatten()
            .map(_annotation_record)
            .flatten()
        )

        return annotations

    @staticmethod
    def _normalize_annotation(annotation):
        """Normalize annotation

        Example:
            >>> data = {"id": "36db0", "annotation_definition": 1,
            ...         "filename": None, "values": [{"a": 10, "b":20}]}
            >>> Captures._normalize_annotation(data)
            # [{"id": "36db0", "annotation_definition": 1,
            #   "filename": None, "values.a": 10, "values.b":20}]
        """
        if not annotation.get(Captures.VALUES_COLUMN):
            return [annotation]
        keys = set(annotation.keys())
        keys.remove(Captures.VALUES_COLUMN)
        # Maybe the json_narmalize was very expensive? Did it somehow not
        # discard
        # the memory used? Did it create extra copy of this json document?
        ann = pd.json_normalize(
            annotation, record_path=Captures.VALUES_COLUMN, meta=list(keys),
            record_prefix=f"{Captures.VALUES_COLUMN}."
        )

        return ann.to_dict(orient="records")

    @profile
    def filter_annotations(self, def_id):
        """Filter annotations by annotation definition id
        :ref:`annotations`

        Args:
            def_id (int): annotation definition id used to filter results

        Returns:
            A Dask dataframe with annotations. Columns: "capture.id", "id",
            "annotation_definition", "values.xyz", "values.abc" ...
            This dataframe is indexed by "capture.id" column to allow quick
            annotation lookup.

        Raises:
            DefinitionIDError: Raised if none of the annotation records in the
                annotations dataframe match the def_id specified as a parameter.
        """
        annotations = (
            self.annotations.filter(
                lambda ann: ann["annotation_definition"] == def_id)
            .map(self._normalize_annotation)
            .flatten()
        )
        # Need to understand if count() is making too much memory overhead?
        # Does this duplicate computation? Maybe move this up after annotations
        # is filtered?
        # Can we do "any" instead of count? Maybe using take
        # try to comment this out and look into dask profiling dashboard?
        if annotations.count().compute() == 0:
            msg = (
                f"Can't find annotations records associated with the given "
                f"definition id: {def_id}."
            )
            raise DefinitionIDError(msg)

        # How costly is this set_index operation? Can we try removing index?
        annotations = (
            annotations.to_dataframe().set_index(self.ANNOTATION_TABLE_INDEX)
        )

        return annotations.compute()
