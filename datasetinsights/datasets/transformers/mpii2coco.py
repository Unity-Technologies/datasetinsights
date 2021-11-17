import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat

from datasetinsights.datasets.transformers.base import DatasetTransformer
from datasetinsights.datasets.transformers.coco import (
    COCO_KEYPOINTS,
    COCO_SKELETON,
)


class MPIItoCOCOTransformer(DatasetTransformer, format="mpii2coco"):
    def __init__(self, data_root, db_type="train", ann_file_path=None):
        self._data_root = Path(data_root)
        self._db_type = db_type

        if ann_file_path:
            self._annotation_file = loadmat(str(ann_file_path))["RELEASE"]
        elif os.path.isfile(
            str(self._data_root / "mpii_human_pose_v1_u12_1.mat")
        ):
            ann_file_path = self._data_root / "mpii_human_pose_v1_u12_1.mat"
            self._annotation_file = loadmat(str(ann_file_path))["RELEASE"]
        else:
            raise ValueError("Annotation file does not exists.")

        self._joint_num = 17

    def execute(self, output, **kwargs):
        self._process_instances(output)

    @staticmethod
    def _copy_image(img_path, dst_dir):
        coco_filename = os.path.basename(img_path)
        dst_path = dst_dir / coco_filename
        shutil.copy2(img_path, dst_path)

    @staticmethod
    def _check_empty(ann_list, name):
        try:
            ann_list[name]
        except ValueError:
            return True
        if len(ann_list[name]) > 0:
            return False
        else:
            return True

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

    @staticmethod
    def _translate_keypoints_in_coco_order(kpt):
        translate_keypoints = {
            0: 16,
            1: 14,
            2: 12,
            3: 11,
            4: 13,
            5: 15,
            6: 0,  # ignore
            7: 1,  # ignore
            8: 2,  # ignore
            9: 3,  # ignore
            10: 10,
            11: 8,
            12: 6,
            13: 5,
            14: 7,
            15: 9,
            16: 4,  # ignore
        }
        keypoints = np.zeros((17, 3))
        for idx, k in enumerate(kpt):
            keypoints[translate_keypoints[idx]] = k

        return keypoints

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

    def _process_joints(self, pid, img_id):
        kpt = np.zeros((self._joint_num, 3))  # xcoord, ycoord, vis
        bbox_kpt = np.zeros((self._joint_num, 3))  # xcoord, ycoord, vis
        # kpt
        ignore_joints = [6, 7, 8, 9]

        # kpt
        annot_joint_num = len(
            self._annotation_file["annolist"][0][0][0][img_id]["annorect"][0][
                pid
            ]["annopoints"]["point"][0][0][0]
        )
        for jid in range(annot_joint_num):
            annot_jid = self._annotation_file["annolist"][0][0][0][img_id][
                "annorect"
            ][0][pid]["annopoints"]["point"][0][0][0][jid]["id"][0][0]

            # for bbox we need all mpii keypoints to determine the bbox
            bbox_kpt[annot_jid][0] = self._annotation_file["annolist"][0][0][0][
                img_id
            ]["annorect"][0][pid]["annopoints"]["point"][0][0][0][jid]["x"][0][
                0
            ]
            bbox_kpt[annot_jid][1] = self._annotation_file["annolist"][0][0][0][
                img_id
            ]["annorect"][0][pid]["annopoints"]["point"][0][0][0][jid]["y"][0][
                0
            ]
            bbox_kpt[annot_jid][2] = 1
            # 1 visibility state because we don't know which ones are occluded

            # to be consistent with coco kpts we don't need certain mpii kpts
            if annot_jid in ignore_joints:
                (kpt[annot_jid][0], kpt[annot_jid][1], kpt[annot_jid][2],) = (
                    0,
                    0,
                    0,
                )
            else:
                kpt[annot_jid][0] = self._annotation_file["annolist"][0][0][0][
                    img_id
                ]["annorect"][0][pid]["annopoints"]["point"][0][0][0][jid]["x"][
                    0
                ][
                    0
                ]
                kpt[annot_jid][1] = self._annotation_file["annolist"][0][0][0][
                    img_id
                ]["annorect"][0][pid]["annopoints"]["point"][0][0][0][jid]["y"][
                    0
                ][
                    0
                ]
                kpt[annot_jid][2] = 1
                # 1 visibility state because we don't know which are occluded

        return bbox_kpt, kpt

    def _process_instances(self, output):
        annotation_output = Path(output) / "annotations"
        annotation_output.mkdir(parents=True, exist_ok=True)

        images_output = Path(output) / "images"
        images_output.mkdir(parents=True, exist_ok=True)

        instances = {
            "info": {"description": "COCO compatible MPII Dataset"},
            "licences": [{"url": "", "id": 1, "name": "default"}],
            "images": [],
            "annotations": [],
            "categories": self._coco_category(),
        }
        aid = 0

        img_num = len(self._annotation_file["annolist"][0][0][0])
        for img_id in range(img_num):

            if (
                (
                    self._db_type == "train"
                    and self._annotation_file["img_train"][0][0][0][img_id] == 1
                )
                or (
                    self._db_type == "test"
                    and self._annotation_file["img_train"][0][0][0][img_id] == 0
                )
            ) and not self._check_empty(
                self._annotation_file["annolist"][0][0][0][img_id], "annorect",
            ):  # any person is annotated

                image_name = str(
                    self._annotation_file["annolist"][0][0][0][img_id]["image"][
                        0
                    ][0][0][0]
                )

                image_path = self._data_root / "images" / image_name
                self._copy_image(img_path=image_path, dst_dir=images_output)
                img = Image.open(image_path)
                w, h = img.size
                img_dict = {
                    "id": img_id,
                    "file_name": image_name,
                    "width": w,
                    "height": h,
                }
                instances["images"].append(img_dict)

                person_num = len(
                    self._annotation_file["annolist"][0][0][0][img_id][
                        "annorect"
                    ][0]
                )  # person_num

                for pid in range(person_num):

                    if not self._check_empty(
                        self._annotation_file["annolist"][0][0][0][img_id][
                            "annorect"
                        ][0][pid],
                        "annopoints",
                    ):  # kpt is annotated
                        bbox_kpt, kpt = self._process_joints(
                            pid=pid, img_id=img_id
                        )

                        # bbox extract from annotated kpt
                        annot_kpt = bbox_kpt[bbox_kpt[:, 2] == 1, :].reshape(
                            -1, 3
                        )
                        bbox = self._extract_bbox_from_kps(kps=annot_kpt)

                        keypoints = self._translate_keypoints_in_coco_order(
                            kpt=kpt
                        )

                        person_dict = {
                            "id": aid,
                            "image_id": img_id,
                            "category_id": 1,
                            "area": bbox[2] * bbox[3],
                            "bbox": bbox.tolist(),
                            "iscrowd": 0,
                            "keypoints": keypoints.reshape(-1).tolist(),
                            "num_keypoints": int(np.sum(keypoints[:, 2] == 1)),
                        }
                        instances["annotations"].append(person_dict)
                        aid += 1

        output_file = annotation_output / "instances.json"
        with open(output_file, "w") as out:
            json.dump(instances, out)
