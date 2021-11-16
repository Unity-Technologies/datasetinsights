import json
import os.path
import shutil
from pathlib import Path

import numpy as np
from coco import COCO_KEYPOINTS, COCO_SKELETON
from PIL import Image
from scipy import io
from tqdm import tqdm

from datasetinsights.datasets.transformers.base import DatasetTransformer

LSPET_JOINTS = (
    "right_ankle",
    "right_knee",
    "right_hip",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_wrist",
    "right_elbow",
    "right_shoulder",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "neck",
    "head_top",
)


class LSPETtoCOCOTransformer(
    DatasetTransformer, format="lspet2coco",
):
    def __init__(self, data_root, ann_file_path=None):
        self._data_root = Path(data_root)
        if ann_file_path:
            self._annotation_file = io.loadmat(str(ann_file_path))["joints"]
        elif os.path.isfile(str(self._data_root / "joints.mat")):
            ann_file_path = self._data_root / "joints.mat"
            self._annotation_file = io.loadmat(str(ann_file_path))["joints"]
        else:
            raise ValueError("Annotation file does not exists.")

    def execute(self, output, **kwargs):
        self._process_instances(output)

    @staticmethod
    def _coco_category():
        category = {
            "supercategory": "person",
            "id": 1,  # to be same as COCO, not using 0
            "name": "person",
            # coco skeleton
            "skeleton": COCO_SKELETON,
            # coco keypoints
            "keypoints": COCO_KEYPOINTS,
        }
        return category

    @staticmethod
    def _extract_bbox_from_kps(kps):
        bbox = np.zeros(4)

        xmin = np.min(kps[:, 0])
        ymin = np.min(kps[:, 1])
        xmax = np.max(kps[:, 0])
        ymax = np.max(kps[:, 1])

        width = xmax - xmin - 1
        height = ymax - ymin - 1

        # corrupted bounding box
        if width <= 0 or height <= 0:
            pass
        # 20% extend
        else:
            bbox[0] = (xmin + xmax) / 2.0 - width / 2 * 1.2
            bbox[1] = (ymin + ymax) / 2.0 - height / 2 * 1.2
            bbox[2] = width * 1.2
            bbox[3] = height * 1.2

        return bbox

    def _process_instances(self, output):
        annotation_output = Path(output) / "annotations"
        annotation_output.mkdir(parents=True, exist_ok=True)

        images_output = Path(output) / "images"
        images_output.mkdir(parents=True, exist_ok=True)

        instances = {
            "info": {"description": "COCO compatible LSPET Dataset"},
            "licences": [{"url": "", "id": 1, "name": "default"}],
            "images": [],
            "annotations": [],
            "categories": self._coco_category(),
        }

        joints_lspet_dict = {}
        for j in range(len(LSPET_JOINTS)):
            joints_lspet_dict[LSPET_JOINTS[j]] = {
                "x": self._annotation_file[j][0],
                "y": self._annotation_file[j][1],
                "v": self._annotation_file[j][2],
            }

        lspet_num_instances = len(joints_lspet_dict["right_ankle"]["x"])

        for i in tqdm(lspet_num_instances):

            str_id = str(i + 1)
            img_id = str_id.zfill(5)
            filename = f"im{img_id}.jpg"
            filepath = self._data_root / "images" / filename
            img = Image.open(filepath)
            coco_filepath = images_output / filename
            shutil.copy2(filepath, coco_filepath)
            w, h = img.size
            img_dict = {
                "id": img_id,
                "file_name": filename,
                "width": w,
                "height": h,
            }
            instances["images"].append(img_dict)

            kpt = np.zeros((len(COCO_KEYPOINTS), 3))  # xcoord, ycoord, vis
            bbox_kpt = np.zeros((len(LSPET_JOINTS), 3))  # xcoord, ycoord, vis

            for j in range(len(COCO_KEYPOINTS)):
                kpt_name = COCO_KEYPOINTS[j]
                if kpt_name in joints_lspet_dict.keys():
                    kpt[j][0] = joints_lspet_dict[kpt_name]["x"][i]
                    kpt[j][1] = joints_lspet_dict[kpt_name]["y"][i]
                    kpt[j][2] = joints_lspet_dict[kpt_name]["v"][i]

                else:
                    kpt[j][0], kpt[j][1], kpt[j][2] = 0, 0, 0

            for j in range(len(LSPET_JOINTS)):
                joint_name = LSPET_JOINTS[j]
                bbox_kpt[j][0] = joints_lspet_dict[joint_name]["x"][i]
                bbox_kpt[j][1] = joints_lspet_dict[joint_name]["y"][i]
                bbox_kpt[j][2] = joints_lspet_dict[joint_name]["v"][i]

            # bbox extract from annotated kps
            annot_kpt = bbox_kpt[bbox_kpt[:, 2] == 1, :].reshape(-1, 3)
            bbox = self._extract_bbox_from_kps(kps=annot_kpt)

            person_dict = {
                "id": img_id,
                "image_id": img_id,
                "category_id": 1,
                "area": bbox[2] * bbox[3],
                "bbox": bbox.tolist(),
                "iscrowd": 0,
                "keypoints": kpt.reshape(-1).tolist(),
                "num_keypoints": int(np.sum(kpt[:, 2] == 1)),
            }
            instances["annotations"].append(person_dict)

        output_file = annotation_output / "instances.json"
        with open(output_file, "w") as out:
            json.dump(instances, out)
