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


def get_img_var_laplacian(img_path: str) -> np.ndarray:
    """
    Computes laplacian and returns the focus measure(ie variance for the image)
    Args:
        img_path (str): Path of image

    Returns:
        Variance of Laplacian of image
    """
    return laplacian_img(img_path).var()


def get_bbox_var_laplacian(
    laplacian: np.ndarray, x: float, y: float, w: float, h: float
) -> np.ndarray:
    """
    Calculates bbox's variance of Laplacian
    Args:
        laplacian (np.ndarray): Laplacian of the image
        x (float): the upper-left coordinate of the bounding box
        y (float): the upper-left coordinate of the bounding box
        w (float): width of bbox
        h (float): height of bbox

    Returns:
        Variance of Laplacian of bbox
    """
    bbox_var = laplacian[int(y) : int(y + h), int(x) : int(x + w)]
    return np.nanvar(bbox_var)


def get_fg_bg_var_laplacian(
    laplacian: np.ndarray, annotations: List[Dict]
) -> Tuple[List, np.ndarray]:
    """
    Calculates foreground and background variance of laplacian of an image
    based on bounding boxes
    Args:
        laplacian (np.ndarray): Laplacian of the image
        annotations (List): List of dictionary of annotations containing bbox
                            information of one image

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
            bbox_var = get_bbox_var_laplacian(img_laplacian, x, y, w, h)
            img_laplacian[int(y) : int(y + h), int(x) : int(x + w)] = np.nan
            bbox_var_lap.append(bbox_var)

    img_var_laplacian = np.nanvar(img_laplacian)

    return bbox_var_lap, img_var_laplacian
