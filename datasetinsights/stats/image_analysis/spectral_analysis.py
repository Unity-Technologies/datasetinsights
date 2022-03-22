import glob

import numpy as np
from PIL import Image
from tqdm import tqdm


def get_psd2d(image: np.ndarray):
    h, w = image.shape
    fourier_image = np.fft.fft2(image)
    N = h * w * 2
    psd2d = (1 / N) * np.abs(fourier_image) ** 2
    psd2d = np.fft.fftshift(psd2d)
    return psd2d


def get_psd1d_az(psd_2d: np.ndarray):
    h = psd_2d.shape[0]
    w = psd_2d.shape[1]
    wc = w // 2
    hc = h // 2

    # create an array of integer radial distances from the center
    y, x = np.ogrid[-h // 2 : h // 2, -w // 2 : w // 2]
    r = np.hypot(x, y).astype(np.int)
    idx = np.arange(0, min(wc, hc))
    psd_1d = _nan_avg(psd_2d, r, index=idx)
    return psd_1d


def _nan_avg(arr, labels, index):
    arr = arr.ravel()
    labels = labels.ravel()
    res = []
    for idx in index:
        pos = np.where(labels == idx)[0]
        sum_ = []
        for p in pos:
            sum_.append(arr[p])
        res.append(np.nanmean(sum_))
    return res


def _pad_image(image, max_h, max_w):
    img = np.empty((max_h, max_w))
    img[:] = np.nan
    hc, wc = img.shape[0] // 2, img.shape[1] // 2
    shc, swc = image.shape[0] // 2, image.shape[1] // 2
    if image.shape[1] % 2 == 0:
        size_w = (wc - swc, wc + swc)
    else:
        size_w = (wc - swc, wc + swc + 1)
    if image.shape[0] % 2 == 0:
        size_h = (hc - shc, hc + shc)
    else:
        size_h = (hc - shc, hc + shc + 1)

    img[size_h[0] : size_h[1], size_w[0] : size_w[1]] = image

    return img


def _load_img(img_path: str):
    img = Image.open(img_path)
    img = img.convert("RGB")
    img = img.convert("L")
    return np.array(img)


def _load_images_from_dir(img_dir: str, img_type: str = "png"):
    image_paths = glob.glob(img_dir + f"/*.{img_type}")
    img_array = []
    max_h, max_w = 0, 0
    for img_path in image_paths:
        img = _load_img(img_path)
        max_h, max_w = max(max_h, img.shape[0]), max(max_w, img.shape[1])
        img_array.append(img)
    return img_array, max_h, max_w


def get_average_psd_1d(img_dir: str, img_type: str = "png"):
    images, max_h, max_w = _load_images_from_dir(img_dir, img_type)
    n = len(images)
    avg_psd_1d = np.zeros(min(max_h, max_w) // 2)

    for image in tqdm(images):
        psd_2d = get_psd2d(image)
        if psd_2d.shape != (max_h, max_w):
            psd_2d = _pad_image(psd_2d, max_h, max_w)

        psd_1d = get_psd1d_az(psd_2d)
        avg_psd_1d += psd_1d

    avg_psd_1d /= n

    return avg_psd_1d
