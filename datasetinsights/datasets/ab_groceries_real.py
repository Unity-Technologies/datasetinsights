
from .groceries_real import GroceriesReal
import datasetinsights.constants as const
from ..io import BBox2D


class ABGroceriesReal(GroceriesReal):

    def __init__(self, *, data_path=const.DEFAULT_DATA_ROOT, split="train", transforms=None, version="v3", target_label=None, **kwargs):
        self.target_label = target_label
        super().__init__(data_path=data_path, split=split, transforms=transforms, version=version, **kwargs)
        self.target_id = next((id for id, label in self.label_mappings.items() if label == target_label), None)
        if self.target_id is None:
            raise ValueError(f"No label {target_label} in dataset. Labels are: {self.label_mappings}")

        self.label_mappings = {1: target_label}

    def __getitem__(self, idx):
        image, bboxes = super().__getitem__(idx)
        bboxes = [BBox2D(x=bbox.x, y=bbox.y, w=bbox.w, h=bbox.h, label=1) for bbox in bboxes if bbox.label == self.target_id ]
        return image, bboxes