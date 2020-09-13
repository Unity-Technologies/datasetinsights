
from .base import Dataset
from .groceries_real import GroceriesReal
import datasetinsights.constants as const
from ..io import BBox2D

class ABGroceriesReal(Dataset):

    def __init__(self, *, data_path=const.DEFAULT_DATA_ROOT, split="train", transforms=None, version="v3", target_label=None, **kwargs):
        self.groceries_real = GroceriesReal(
            data_path=data_path, split=split, transforms=None, version=version, **kwargs
        )
        self.transforms = transforms
        self.target_label = target_label
        self.target_id = next((id for id, label in self.groceries_real.label_mappings.items() if label == target_label), None)
        if self.target_id is None:
            raise ValueError(f"No label {target_label} in dataset. Labels are: {self.groceries_real.label_mappings}")

        self.label_mappings = {1: target_label}

        self._init_images_with_data()

    def __getitem__(self, idx):
        groceries_idx = self.image_indices_with_data[idx]
        image, bboxes = self.get_image_and_bboxes_for_index(groceries_idx)

        if self.transforms:
            image, bboxes = self.transforms(image, bboxes)

        return image, bboxes

    def __len__(self):
        return self.image_indices_with_data.__len__()

    def get_image_and_bboxes_for_index(self, idx):
        image, bboxes = self.groceries_real.__getitem__(idx)
        for bbox in bboxes:
            if isinstance(bbox, str):
                raise ValueError(f"bbox for index {idx} is a string: {bbox}")

        bboxes = [BBox2D(x=bbox.x, y=bbox.y, w=bbox.w, h=bbox.h, label=1) for bbox in bboxes if bbox.label == self.target_id ]
        return image, bboxes

    def _init_images_with_data(self):
        self.image_indices_with_data = []
        for i in range(self.groceries_real.__len__()):
            image, bboxes = self.get_image_and_bboxes_for_index(i)
            if any(bboxes):
                self.image_indices_with_data.append(i)