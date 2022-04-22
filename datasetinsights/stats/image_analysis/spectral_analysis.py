import glob

import numpy as np
from PIL import Image
from scipy import ndimage
from tqdm import tqdm


def get_psd2d(image: np.ndarray) -> np.ndarray:
    """
    Args:
        image (np.ndarray): Grayscale Image

    Returns:
        np.ndarray: 2D PSD of the image
    """
    h, w = image.shape
    fourier_image = np.fft.fft2(image)
    N = h * w * 2
    psd2d = (1 / N) * np.abs(fourier_image) ** 2
    psd2d = np.fft.fftshift(psd2d)
    return psd2d


def get_psd1d(psd_2d: np.ndarray) -> np.ndarray:
    """
    Args:
        psd_2d (np.ndarray): 2D PSD of the image

    Returns:
        np.ndarray: 1D PSD of the given 2D PSD
    """
    h = psd_2d.shape[0]
    w = psd_2d.shape[1]
    wc = w // 2
    hc = h // 2

    # create an array of integer radial distances from the center
    y, x = np.ogrid[-h // 2 : h // 2, -w // 2 : w // 2]
    r = np.hypot(x, y).astype(int)
    idx = np.arange(0, min(wc, hc))
    psd_1d = ndimage.sum(psd_2d, r, index=idx)
    return psd_1d


def _load_img(img_path: str):
    img = Image.open(img_path)
    img = img.convert("RGB")
    img = img.convert("L")
    return np.array(img)


def _load_images_from_dir(img_dir: str, img_type: str = "png"):
    image_paths = glob.glob(img_dir + f"/*.{img_type}")
    img_array = []
    for img_path in image_paths:
        img = _load_img(img_path)
        img_array.append(img)
    return img_array


def get_average_psd_1d(img_dir: str, img_type: str = "png"):
    """
    Get average PSD of entire dataset.
    Args:
        img_dir (str): Path of image directory
        img_type (str): Image tpye (PNG, JPG, etc)

    Returns:
        avg_psd_1d (np.ndarray): Avg PSD 1D
        std_psd_1d (np.ndarray): Standard deviation of PSD

    """
    images = _load_images_from_dir(img_dir, img_type)
    total_psd_1d = []
    max_len = float("-inf")

    for image in tqdm(images):
        psd_2d = get_psd2d(image)
        psd_1d = get_psd1d(psd_2d)
        max_len = max(max_len, len(psd_1d))
        total_psd_1d.append(psd_1d)

    for i in range(len(total_psd_1d)):
        if len(total_psd_1d[i]) < max_len:
            _len = max_len - len(total_psd_1d[i])
            nan_arr = np.empty(_len)
            nan_arr[:] = np.nan
            total_psd_1d[i] = np.append(total_psd_1d[i], nan_arr)

    total_psd_1d = np.asarray(total_psd_1d, dtype=float)

    avg_psd_1d = np.nanmean(total_psd_1d, axis=0)
    std_psd_1d = np.nanstd(total_psd_1d, axis=0)

    return avg_psd_1d, std_psd_1d
