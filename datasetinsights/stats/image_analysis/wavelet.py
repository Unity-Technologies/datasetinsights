import glob
import random

import numpy as np
import pywt
from PIL import Image
from tqdm import tqdm


def get_wt_coeffs_var(img_dir: str, img_type: str = "png", num_img=None):
    """

    Args:
        img_dir (str): Path of image directory
        img_type (str): Image tpye (PNG, JPG, etc)
        num_img (int): Number of images to use for the calculation

    Returns:
        List of variance of Horizontal, Vertical and Diagonal details

    """
    images = glob.glob(img_dir + f"/*.{img_type}")

    if num_img and num_img < len(images):
        images = random.sample(images, num_img)

    horizontal_coeff, vertical_coeff, diagonal_coeff = [], [], []

    for img in tqdm(images):
        im = Image.open(img).convert("L")
        _, (cH, cV, cD) = pywt.dwt2(im, "haar", mode="periodization")
        horizontal_coeff.append(np.array(cH).var())
        vertical_coeff.append(np.array(cV).var())
        diagonal_coeff.append(np.array(cD).var())

    return horizontal_coeff, vertical_coeff, diagonal_coeff
