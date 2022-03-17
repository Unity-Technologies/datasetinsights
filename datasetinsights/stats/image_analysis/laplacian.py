from typing import Dict, List, Tuple

import cv2
import numpy as np


def laplacian_img(img_path: str) -> np.ndarray:
    """
    Converts image to grayscale, computes laplacian and returns it.
    Args:
        img_path (str): Path of image

    Returns:
        np.ndarray: numpy array of Laplacian of the image
    """
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = laplacian.astype("float")
    return laplacian


def get_bbox_var_laplacian(
    laplacian: np.ndarray, x: int, y: int, w: int, h: int
) -> np.ndarray:
    """
    Calculates bbox's variance of Laplacian
    Args:
        laplacian (np.ndarray): Laplacian of the image
        x (int): the upper-left coordinate of the bounding box
        y (int): the upper-left coordinate of the bounding box
        w (int): width of bbox
        h (int): height of bbox

    Returns:
        Variance of Laplacian of bbox
    """
    bbox_var = laplacian[y : y + h, x : x + w]
    return np.nanvar(bbox_var)


def get_bbox_fg_bg_var_laplacian(
    laplacian: np.ndarray, annotations: List[Dict]
) -> Tuple[List, np.ndarray]:
    """
    Calculates foreground and background variance of laplacian of an image
    based on bounding boxes
    Args:
        laplacian (np.ndarray): Laplacian of the image
        annotations (List): List of dictionary of annotations containing bbox
                            information of the given image laplacian

    Returns:
        bbox_var_lap (List): List of variance of laplacian of all bbox in the
        image
        img_var_laplacian (np.ndarray): Variance of Laplacian of background
        of the image

    """
    bbox_var_lap = []
    img_laplacian = laplacian

    for ann in annotations:
        x, y, w, h = ann["bbox"]
        bbox_area = w * h
        if bbox_area >= 1200:  # ignoring small bbox sizes
            bbox_var = get_bbox_var_laplacian(
                img_laplacian, int(x), int(y), int(w), int(h)
            )
            img_laplacian[int(y) : int(y + h), int(x) : int(x + w)] = np.nan
            bbox_var_lap.append(bbox_var)

    img_var_laplacian = np.nanvar(img_laplacian)

    return bbox_var_lap, img_var_laplacian


def get_final_mask(masks: List[np.ndarray]) -> np.ndarray:
    """
    Get one masks from multiple mask of an image
    Args:
        masks (List[np.ndarray]): List of binary masks of an image

    Returns:
        final_mask = Final binary mask representing union of all masks of an
        images
    """
    final_mask = np.zeros_like(masks[0])
    for mask in masks:
        final_mask = np.bitwise_or(final_mask, mask)
    return final_mask


def get_seg_fg_bg_var_laplacian(
    laplacian: np.ndarray, final_mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates foreground and background variance of laplacian of an image
    based on segmentation information
    Args:
        laplacian (np.ndarray): Laplacian of the image
        final_mask (np.ndarray): Binary mask of the image in which 1 is
        instances of the image

    Returns:
        fg_var_lap = Foreground var of laplacian
        bg_var_lap = Background var of laplacian

    """
    fg = np.where(final_mask == 0, laplacian, np.nan)
    bg = np.where(final_mask == 1, laplacian, np.nan)
    fg_var_lap = np.nanvar(fg)
    bg_var_lap = np.nanvar(bg)

    return fg_var_lap, bg_var_lap
