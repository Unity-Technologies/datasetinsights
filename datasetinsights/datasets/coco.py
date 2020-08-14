import json
import logging
import os
import random
import zipfile
from os import makedirs
from os.path import isdir, join
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torchvision
from PIL.Image import Image
from pycocotools.coco import COCO

import datasetinsights.constants as const
from datasetinsights.io.bbox import BBox2D
from datasetinsights.storage.gcs import GCSClient

from .base import Dataset

ANNOTATION_FILE_TEMPLATE = "{}_{}2017.json"
COCO_GCS_PATH = "data/coco"
COCO_LOCAL_PATH = "coco"
logger = logging.getLogger(__name__)


def _coco_remove_images_without_annotations(dataset):
    """

    Args:
        dataset (torchvision.datasets.CocoDetection):

    Returns (torch.utils.data.Subset): filters dataset to exclude examples
    which either have no bounding boxes or have an invalid bounding box (a
    bounding box is invalid if it's height or width is <1).

    """

    def _has_any_empty_box(anno):
        return any(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

    def _has_valid_annotation(anno):
        # if it's empty, there is no annotation
        if len(anno) == 0:
            return False
        if _has_any_empty_box(anno):
            return False
        return True

    ids = []
    for ds_idx, img_id in enumerate(dataset.ids):
        ann_ids = dataset.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = dataset.coco.loadAnns(ann_ids)
        if _has_valid_annotation(anno):
            ids.append(ds_idx)
    dataset = torch.utils.data.Subset(dataset, ids)
    return dataset


def convert_coco2canonical(coco_annotation):
    """
    convert from a tuple of image and coco style dictionary describing the
    bboxes to a tuple of image, List of BBox2D
    Args:
        coco_annotation (tuple): image and coco style dictionary

    Returns: a tuple of image, List of BBox2D

    """
    image, targets = coco_annotation
    all_bboxes = []
    for t in targets:
        label = t["category_id"]
        bbox = t["bbox"]
        b = BBox2D(x=bbox[0], y=bbox[1], w=bbox[2], h=bbox[3], label=label)
        all_bboxes.append(b)
    return image, all_bboxes


class CocoDetection(Dataset):
    """
    http://cocodataset.org/#detection-2019
    """

    def __init__(
        self,
        *,
        data_root=const.DEFAULT_DATA_ROOT,
        split="train",
        transforms=None,
        remove_examples_without_boxes=True,
        **kwargs,
    ):
        # todo add test split
        self.split = split
        self.root = os.path.join(data_root, COCO_LOCAL_PATH)
        self.download()
        self.coco = self._get_coco(root=self.root, image_set=split)
        if remove_examples_without_boxes:
            self.coco = _coco_remove_images_without_annotations(
                dataset=self.coco
            )
        self.transforms = transforms

    def __getitem__(self, idx) -> Tuple[Image, List[BBox2D]]:
        """
        Args:
            idx:

        Returns: Image with list of bounding boxes found inside the image

        """
        image, target = convert_coco2canonical(self.coco[idx])
        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.coco)

    def _get_coco(self, root, image_set, mode="instances"):
        PATHS = {
            "train": (
                "train2017",
                os.path.join(
                    "annotations",
                    ANNOTATION_FILE_TEMPLATE.format(mode, "train"),
                ),
            ),
            "val": (
                "val2017",
                os.path.join(
                    "annotations", ANNOTATION_FILE_TEMPLATE.format(mode, "val")
                ),
            ),
        }
        img_folder, ann_file = PATHS[image_set]
        img_folder = os.path.join(root, img_folder)
        ann_file = os.path.join(root, ann_file)
        coco = torchvision.datasets.CocoDetection(img_folder, ann_file)
        return coco

    def _get_local_annotations_zip(self):
        return os.path.join(self.root, "annotations_trainval2017.zip")

    def _get_local_images_zip(self):
        return os.path.join(self.root, f"{self.split}2017.zip")

    def download(self, cloud_path=COCO_GCS_PATH):
        path = Path(self.root)
        path.mkdir(parents=True, exist_ok=True)

        client = GCSClient()
        annotations_zip_gcs = f"{cloud_path}/annotations_trainval2017.zip"
        annotations_zip_2017 = self._get_local_annotations_zip()
        logger.info(f"checking for local copy of data")
        if not os.path.exists(annotations_zip_2017):
            logger.info(f"no annotations zip file found, will download.")
            client.download(
                bucket_name=const.GCS_BUCKET,
                object_key=annotations_zip_gcs,
                localfile=annotations_zip_2017,
            )
            with zipfile.ZipFile(annotations_zip_2017, "r") as zip_dir:
                zip_dir.extractall(self.root)
        images_local = self._get_local_images_zip()
        images_gcs = f"{cloud_path}/{self.split}2017.zip"
        if not os.path.exists(images_local):
            logger.info(
                f"no zip file for images for {self.split} found,"
                f" will download"
            )
            client.download(
                bucket_name=const.GCS_BUCKET,
                object_key=images_gcs,
                localfile=images_local,
            )
            with zipfile.ZipFile(images_local, "r") as zip_dir:
                zip_dir.extractall(self.root)


class CocoTracking(Dataset):
    def __init__(self, anchors, anchor_target, config, transforms):
        global logger
        logger = logging.getLogger("global")

        self.split = "train"
        self.download()
        self.crop()
        self.generate_json_coco()
        self.anchors = anchors
        self.template_size = 127
        self.origin_size = 127
        self.search_size = 255
        # DO NOT change. TODO: Write tests for this
        self.size = 25
        self.base_size = 0
        self.crop_size = 255
        self.config = config

        self.dset = "CocoTracking"

        self.alldata = []
        self.pick = None

        # This will be calculated in the read function (populate)
        self.num = 0

        self.get_transforms = transforms

        # This generates the anchor boxes
        self.anchors.generate_all_anchors(
            im_c=self.search_size // 2, size=self.size
        )
        # Assigns the class information for anchor boxes forg/backg
        self.anchor_target = anchor_target

        self.alldata = []
        self.populate(self.config)
        self.shuffle()

    def __len__(self):
        return self.num

    def _get_local_annotations_zip(self):
        return os.path.join(self.root, "annotations_trainval2017.zip")

    def _get_local_images_zip(self):
        return os.path.join(self.root, f"{self.split}2017.zip")

    def download(self, cloud_path=COCO_GCS_PATH):
        data_root = "./datasetinsights/data"
        self.root = os.path.join(data_root, COCO_LOCAL_PATH)
        path = Path(self.root)
        path.mkdir(parents=True, exist_ok=True)
        client = GCSClient()
        annotations_zip_gcs = f"{cloud_path}/annotations_trainval2017.zip"
        annotations_zip_2017 = self._get_local_annotations_zip()
        logger.info(f"checking for local copy of data")
        if not os.path.exists(annotations_zip_2017):
            logger.info(f"no annotations zip file found, will download.")
            client.download(
                bucket_name=const.GCS_BUCKET,
                object_key=annotations_zip_gcs,
                localfile=annotations_zip_2017,
            )
            with zipfile.ZipFile(annotations_zip_2017, "r") as zip_dir:
                zip_dir.extractall(self.root)
        images_local = self._get_local_images_zip()
        images_gcs = f"{cloud_path}/{self.split}2017.zip"
        if not os.path.exists(images_local):
            logger.info(
                f"no zip file for images for {self.split} found,"
                f" will download"
            )
            client.download(
                bucket_name=const.GCS_BUCKET,
                object_key=images_gcs,
                localfile=images_local,
            )
            with zipfile.ZipFile(images_local, "r") as zip_dir:
                zip_dir.extractall(self.root)

    def crop_hwc(self, image, bbox, out_sz, padding=(0, 0, 0)):
        a = (out_sz - 1) / (bbox[2] - bbox[0])
        b = (out_sz - 1) / (bbox[3] - bbox[1])
        c = -a * bbox[0]
        d = -b * bbox[1]
        mapping = np.array([[a, 0, c], [0, b, d]]).astype(np.float)
        crop = cv2.warpAffine(
            image,
            mapping,
            (out_sz, out_sz),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=padding,
        )
        return crop

    def pos_s_2_bbox(self, pos, s):
        return [pos[0] - s / 2, pos[1] - s / 2, pos[0] + s / 2, pos[1] + s / 2]

    def crop_like_SiamFCx(
        self,
        image,
        bbox,
        exemplar_size=127,
        context_amount=0.5,
        search_size=255,
        padding=(0, 0, 0),
    ):
        target_pos = [(bbox[2] + bbox[0]) / 2.0, (bbox[3] + bbox[1]) / 2.0]
        target_size = [bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1]
        wc_z = target_size[1] + context_amount * sum(target_size)
        hc_z = target_size[0] + context_amount * sum(target_size)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        d_search = (search_size - exemplar_size) / 2
        pad = d_search / scale_z
        s_x = s_z + 2 * pad

        x = self.crop_hwc(
            image, self.pos_s_2_bbox(target_pos, s_x), search_size, padding
        )
        return x

    def crop_img(
        self,
        img,
        anns,
        set_crop_base_path,
        set_img_base_path,
        exemplar_size=127,
        context_amount=0.5,
        search_size=511,
        enable_mask=True,
    ):
        frame_crop_base_path = join(
            set_crop_base_path, img["file_name"].split("/")[-1].split(".")[0]
        )
        if not isdir(frame_crop_base_path):
            makedirs(frame_crop_base_path)

        im = cv2.imread("{}/{}".format(set_img_base_path, img["file_name"]))
        avg_chans = np.mean(im, axis=(0, 1))
        for track_id, ann in enumerate(anns):
            rect = ann["bbox"]
            if rect[2] <= 0 or rect[3] <= 0:
                continue
            bbox = [
                rect[0],
                rect[1],
                rect[0] + rect[2] - 1,
                rect[1] + rect[3] - 1,
            ]

            x = self.crop_like_SiamFCx(
                im,
                bbox,
                exemplar_size=exemplar_size,
                context_amount=context_amount,
                search_size=search_size,
                padding=avg_chans,
            )
            cv2.imwrite(
                join(
                    frame_crop_base_path,
                    "{:06d}.{:02d}.x.jpg".format(0, track_id),
                ),
                x,
            )

            if enable_mask:
                im_mask = coco.annToMask(ann).astype(np.float32)
                x = (
                    self.crop_like_SiamFCx(
                        im_mask,
                        bbox,
                        exemplar_size=exemplar_size,
                        context_amount=context_amount,
                        search_size=search_size,
                    )
                    > 0.5
                ).astype(np.uint8) * 255
                cv2.imwrite(
                    join(
                        frame_crop_base_path,
                        "{:06d}.{:02d}.m.png".format(0, track_id),
                    ),
                    x,
                )
                # print("Done: ",'{:06d}.{:02d}.m.png'.format(0, track_id))

    def generate_json_coco(self):
        dataDir = self.root
        for data_subset in ["train2017"]:
            dataset = dict()
            annFile = "{}/annotations/instances_{}.json".format(
                dataDir, data_subset
            )
            coco = COCO(annFile)
            n_imgs = len(coco.imgs)
            for n, img_id in enumerate(coco.imgs):
                print(
                    "subset: {} image id: {:04d} / {:04d}".format(
                        data_subset, n, n_imgs
                    )
                )
                img = coco.loadImgs(img_id)[0]
                annIds = coco.getAnnIds(imgIds=img["id"], iscrowd=None)
                anns = coco.loadAnns(annIds)
                crop_base_path = join(
                    data_subset, img["file_name"].split("/")[-1].split(".")[0]
                )

                if len(anns) > 0:
                    dataset[crop_base_path] = dict()

                for track_id, ann in enumerate(anns):
                    rect = ann["bbox"]
                    if rect[2] <= 0 or rect[3] <= 0:  # lead nan error in cls.
                        continue
                    bbox = [
                        rect[0],
                        rect[1],
                        rect[0] + rect[2] - 1,
                        rect[1] + rect[3] - 1,
                    ]  # x1,y1,x2,y2

                    dataset[crop_base_path]["{:02d}".format(track_id)] = {
                        "000000": bbox
                    }

            print("save json (dataset), please wait 20 seconds~")
            json.dump(
                dataset,
                open(dataDir + "/{}.json".format(data_subset), "w"),
                indent=4,
                sort_keys=True,
            )
            print("done!")

    def crop(
        self,
        exemplar_size=127,
        context_amount=0.5,
        search_size=511,
        enable_mask=True,
        num_threads=24,
    ):
        global coco  # will used for generate mask
        data_dir = self.root
        crop_path = self.root + "/crop{:d}".format(search_size)
        if not isdir(crop_path):
            os.mkdir(crop_path)

        for data_subset in ["train2017"]:
            set_crop_base_path = join(crop_path, data_subset)
            set_img_base_path = join(data_dir, data_subset)

            anno_file = "{}/annotations/instances_{}.json".format(
                data_dir, data_subset
            )
            coco = COCO(anno_file)
            n_imgs = len(coco.imgs)

            for cnt, i in enumerate(coco.imgs):
                logger.info(str(cnt) + "/" + str(n_imgs))
                self.crop_img(
                    coco.loadImgs(i)[0],
                    coco.loadAnns(coco.getAnnIds(imgIds=i, iscrowd=None)),
                    set_crop_base_path,
                    set_img_base_path,
                    exemplar_size,
                    context_amount,
                    search_size,
                    enable_mask,
                )

            # TODO: Figure out why async process pool is not working.
            # github.com/foolwood/SiamMask/blob/master/data/coco/par_crop.py
            # Single thread takes ~1.5 hours to process

            # with futures.ProcessPoolExecutor(max_workers=num_threads)
            # as executor:
            #     fs = [executor.submit(crop_img, coco.loadImgs(id)[0],
            #                           coco.loadAnns(
            #                           coco.getAnnIds(imgIds=id,
            #                           iscrowd=None)),
            #                           set_crop_base_path,
            #                           set_img_base_path,
            #                           exemplar_size,
            #                           context_amount, search_size,
            #                           enable_mask) for id in coco.imgs]

            #     for i, f in enumerate(futures.as_completed(fs)):
            #         printProgress(i, n_imgs,
            #         prefix=data_subset, suffix='Done ', barLength=40)

    # Apply all transforms and return object
    def __getitem__(self, index):
        index = self.pick[index]
        # TODO: put both negs in config files
        self.neg = 0.25
        self.inner_neg = 0.5

        # neg means the probability by which a negative pair is selected
        # inner_neg is inside a neg inside subdataset
        #  <neg value = more robust training <<neg can disrupt training
        neg = self.neg and self.neg > random.random()
        neg = False

        if neg:
            template = self.get_random_target(index)
            if self.inner_neg and self.inner_neg > random.random():
                search = self.get_random_target()
            else:
                search = random.choice(self.alldata).get_random_target()
        else:
            template, search = self.get_positive_pair(index)

        # Read images
        # Read Images returned from get_random
        template_image, scale_z = self.imread(template[0])
        search_image, scale_x = self.imread(search[0])
        # Read the mask if the dataset has a mask
        # If mask doesnt exist only return a zero np array
        if self.has_mask and not neg:
            search_mask = (cv2.imread(search[2], 0)).astype(np.float32)
        else:
            search_mask = np.ones(search_image.shape[:2], dtype=np.float32)

        blur = 1
        gray = 1

        template_transforms = self.get_transforms(
            center_crop=(True, self.template_size),
            blur=(True, blur),
            gray=(False, gray),
            shift=(True, 4),
            scale=(True, 0.05),
            resize=(True,),
            flip=(True,),
        )

        search_transforms = self.get_transforms(
            center_crop=(True, self.crop_size),
            blur=(True, blur),
            gray=(False, gray),
            shift=(True, 64),
            scale=(True, 0.18),
            resize=(True,),
            flip=(True,),
        )

        template, _, _, _ = template_transforms(
            template_image, template[1], self.template_size, None
        )
        search, bbox, mask, bbox_orig = search_transforms(
            search_image, search[1], self.search_size, search_mask
        )

        neg = False
        cls, delta, delta_weight = self.anchor_target(
            self.anchors, bbox, self.size, neg
        )

        # Change this if we want to train using any dataset
        # which has no masks like vid

        mask_weight = cls.max(axis=0, keepdims=True)
        # if dataset.has_mask and not neg:
        # else:
        # mask_weight=np.zeros([1,cls.shape[1],cls.shape[2]],dtype=np.float32)

        # Changing ordering of channels for pytorch format
        template, search = map(
            lambda x: np.transpose(x, (2, 0, 1)).astype(np.float32),
            [template, search],
        )

        # To get into Pytorch format
        mask = (np.expand_dims(mask, axis=0) > 0.5) * 2 - 1  # 1*H*W

        return (
            template,
            search,
            cls,
            delta,
            delta_weight,
            np.array(
                [bbox.x, bbox.y, bbox.x + bbox.w, bbox.y + bbox.h], np.float32
            ),
            np.array(mask, np.float32),
            np.array(mask_weight, np.float32),
        )

    def imread(self, path):
        img = cv2.imread(path)

        if self.origin_size == self.template_size:
            return img, 1.0

        def map_size(exe, size):
            return int(
                round(((exe + 1) / (self.origin_size + 1) * (size + 1) - 1))
            )

        nsize = map_size(self.template_size, img.shape[1])

        img = cv2.resize(img, (nsize, nsize))

        return img, nsize / img.shape[1]

    """
        Populate function is used to read the dataset and load from the
        json file created
    """

    def populate(self, cfg):
        dset_index = cfg["dataset"]["name"].index(self.dset)
        root = cfg["dataset"]["dataroot"][dset_index]
        anno = cfg["dataset"]["annot_file"][dset_index]

        with open(anno) as fin:
            logger.info("loading " + anno)
            self.labels = self.filter_zero(json.load(fin), self.dset, cfg)

            def isint(x):
                try:
                    int(x)
                    return True
                except ValueError:
                    return False

            # add frames args into labels
            to_del = []
            for video in self.labels:
                for track in self.labels[video]:
                    frames = self.labels[video][track]
                    frames = list(
                        map(int, filter(lambda x: isint(x), frames.keys()))
                    )
                    frames.sort()
                    self.labels[video][track]["frames"] = frames
                    if len(frames) <= 0:
                        logger.info(
                            "warning {}/{} has no frames.".format(video, track)
                        )
                        to_del.append((video, track))

            # delete tracks with no frames
            for video, track in to_del:
                del self.labels[video][track]

            # delete videos with no valid track
            to_del = []
            for video in self.labels:
                if len(self.labels[video]) <= 0:
                    logger.info("warning {} has no tracks".format(video))
                    to_del.append(video)

            for video in to_del:
                del self.labels[video]

            self.videos = list(self.labels.keys())
            # print("Total vids:",len(self.videos))
            # exit(0)
            logger.info(anno + " loaded.")

        self.root = root
        self.start = 0
        self.num = len(self.labels)
        self.num_use = cfg["dataset"]["num_use"][0]
        if self.num_use == -1:
            self.num_use = self.num
        # Frame range is x +/- 100 range.
        # If kept high, the object might disappear from frame
        self.frame_range = 100
        self.path_format = "{}.{}.{}.jpg"
        self.mask_format = "{}.{}.m.png"
        self.pick = []

        self.num_use = int(self.num_use)

        # self.has_mask = self.mark in ["coco", "ytb_vos"]
        # Kept here for generic purpose. Whenever new class is made-change this
        self.has_mask = True

        """
        'Pick' is a list of index mappings from real to shuffled indices
        To get consistent results the random variable with fixed seed can be set
        """
        # self.shuffle()
        # print(self.__dict__.keys())

    """
    Function to shuffle the dataset as a whole.
    Pick is created here
    'Pick' is a list of index mappings from real to shuffled indices
    To get consistent results the random variable with fixed seed can be set
    """

    def shuffle(self):
        lists = list(range(self.start, self.start + self.num))
        m = 0
        pick = []
        while m < self.num_use:
            random.Random().shuffle(lists)
            pick += lists
            m += self.num

        self.pick = pick[: self.num_use]
        return self.pick

    """
    Function to remove the frames with 0 w/h
    """

    def filter_zero(self, anno, dset, cfg):
        name = dset
        # cfg.get('mark', '')
        # print(len(anno))
        out = {}
        tot = 0
        new = 0
        zero = 0

        for video, tracks in anno.items():
            new_tracks = {}
            for trk, frames in tracks.items():
                new_frames = {}
                for frm, bbox in frames.items():
                    tot += 1
                    if len(bbox) == 4:
                        x1, y1, x2, y2 = bbox
                        w, h = x2 - x1, y2 - y1
                    else:
                        w, h = bbox
                    if w == 0 or h == 0:
                        logger.info(
                            "Error, {name} {video} {trk} {bbox}".format(
                                **locals()
                            )
                        )
                        zero += 1
                        continue
                    new += 1
                    new_frames[frm] = bbox

                if len(new_frames) > 0:
                    new_tracks[trk] = new_frames

            if len(new_tracks) > 0:
                out[video] = new_tracks

        return out

    def get_image_anno(self, video, track, frame):
        frame = "{:06d}".format(frame)
        image_path = join(
            self.root, video, self.path_format.format(frame, track, "x")
        )
        image_anno = self.labels[video][track][frame]

        mask_path = join(
            self.root, video, self.mask_format.format(frame, track)
        )

        return image_path, image_anno, mask_path

    def get_positive_pair(self, index):
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = random.choice(list(video.keys()))
        track_info = video[track]

        frames = track_info["frames"]

        if "hard" not in track_info:
            template_frame = random.randint(0, len(frames) - 1)

            left = max(template_frame - self.frame_range, 0)
            right = min(template_frame + self.frame_range, len(frames) - 1) + 1
            search_range = frames[left:right]
            template_frame = frames[template_frame]
            search_frame = random.choice(search_range)
        else:
            search_frame = random.choice(track_info["hard"])
            left = max(search_frame - self.frame_range, 0)
            right = (
                min(search_frame + self.frame_range, len(frames) - 1) + 1
            )  # python [left:right+1) = [left:right]
            template_range = frames[left:right]
            template_frame = random.choice(template_range)
            search_frame = frames[search_frame]

        return (
            self.get_image_anno(video_name, track, template_frame),
            self.get_image_anno(video_name, track, search_frame),
        )

    def get_random_target(self, index=-1):
        if index == -1:
            index = random.randint(0, self.num - 1)
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = random.choice(list(video.keys()))
        track_info = video[track]

        frames = track_info["frames"]
        frame = random.choice(frames)

        return self.get_image_anno(video_name, track, frame)
