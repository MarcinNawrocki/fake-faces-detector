import typing as t

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import convolve
from skimage.color import rgb2hsv, rgb2ycbcr
from skimage.feature import greycomatrix


def all_colorspaces_from_rgb(np_rgb_img: np.ndarray) -> np.ndarray:
    """Generates numpy array containing 9 color componenets from RGB image with shape (x,y,3). 
    Color componenents order is R,G,B,H,S,V,Y,Cb,Cr


    Args:
        np_rgb_img (np.ndarray): rgb image

    Raises:
        ValueError: wrong image shape

    Returns:
        np.ndarray: array containing nine color componenets with shape (x,y,9)
    """
    if np_rgb_img.shape[2] < 3 or len(np_rgb_img.shape) != 3:
        raise ValueError("Bad shape of input image")
    np_img_hsv = rgb2hsv(np_rgb_img)
    np_img_ycbcr = rgb2ycbcr(np_rgb_img) / 255
    # (x,y,9) shape insted of (x,y,3)
    new_shape = list(np_rgb_img.shape[:2])+[9]
    np_img_all_colors = np.empty(new_shape)
    np_img_all_colors[:, :, :3] = np_rgb_img
    np_img_all_colors[:, :, 3:6] = np_img_hsv
    np_img_all_colors[:, :, 6:] = np_img_ycbcr
    return np_img_all_colors


def comatrix_from_image(np_img: np.ndarray, distances: t.List[int], angles: t.List[float]) -> np.ndarray:
    """Generate comatrixes from image for specified distances and angles. One comatrix for each combination

    Args:
        np_img (np.ndarray): input image
        distances (t.List[int]): distances for comatrix calculation
        angles (t.List[float]): angle for comatrix calculation

    Raises:
        ValueError: len(np_img.shape) other than 2 or 3

    Returns:
        np.ndarray: array containing comatrixes for each distance and angle combination
    """
    np_comatrix = np.empty(np_img.shape+(len(distances), len(angles)))
    if len(np_img.shape) == 3:
        for i in range(np_img.shape[-1]):
            np_comatrix[:, :, i] = greycomatrix(
                np_img[:, :, i], distances, angles)
    elif len(np_img.shape) == 2:
        np_comatrix = greycomatrix(np_img, distances, angles)
    else:
        raise ValueError('Bad shape of the image')
    return np_comatrix


def calculate_difference_image(np_img: np.ndarray) -> np.ndarray:
    """Calculate difference image that is convolution with (1,-1) filter

    Args:
        np_img (np.ndarray): input image, should be image with integer pixels in range 0-255

    Raises:
        ValueError: len(np_img.shape) other than 2 or 3

    Returns:
        np.ndarray: calculated diffference image
    """
    np_filter = np.array((1, -1)).reshape(1, 2)
    np_img = np_img.astype(np.int16)
    np_diff_img = np.empty(np_img.shape, dtype=np.int16)
    if len(np_img.shape) == 3:
        for i in range(np_img.shape[-1]):
            np_diff_img[:, :, i] = convolve(np_img[:, :, i], np_filter)
    elif len(np_img.shape) == 2:
        np_diff_img = convolve(np_img, np_filter)
    else:
        raise ValueError('Bad shape of the image')
    return np_diff_img


def hist_peek_point(np_img: np.ndarray, bins=511, hist_range=(-255, 256)) -> t.Tuple[float, int]:
    """Calculate histogram peek point

    Args:
        np_img (np.ndarray): input image to calculate histograms 
        bins (int, optional): number of histogram intervals. Defaults to 511.
        hist_range (tuple, optional): image pixels range. Defaults to (-255, 256).

    Returns:
        t.Tuple[float, int]: peek point coordinates (pixel value, number of occurences)
    """
    hist, bins = np.histogram(np_img, density=True,
                              bins=bins, range=hist_range)
    y = hist.max()
    idx = int(np.argwhere(hist == y))
    x = int(bins[idx])
    return x, y
