import typing as t

import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2hsv, rgb2ycbcr



def all_colorspaces_from_rgb(np_rgb_img: np.ndarray, type='float') -> np.ndarray:
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

    np_img_hsv = rgb_to_hsv(np_rgb_img, type=type)
    np_img_ycbcr = rgb_to_ycbcr(np_rgb_img, type=type)
    # (x,y,9) shape insted of (x,y,3)
    new_shape = list(np_rgb_img.shape[:2])+[9]
    new_type = np.float64 if type=='float' else np.uint8
    np_img_all_colors = np.empty(new_shape, dtype=new_type)
    np_img_all_colors[:, :, :3] = np_rgb_img
    np_img_all_colors[:, :, 3:6] = np_img_hsv
    np_img_all_colors[:, :, 6:] = np_img_ycbcr
    return np_img_all_colors

def rgb_to_hsv(np_rgb_img: np.ndarray, type='float'):
    if type == 'float':
        np_img_hsv = rgb2hsv(np_rgb_img)
    elif type == 'int':
        np_img_hsv = (rgb2hsv(np_rgb_img)*255).astype(np.uint8)
    else:
        raise ValueError("Bad type specified")
        
    return np_img_hsv

def rgb_to_ycbcr(np_rgb_img: np.ndarray, type='float'):
    if type == 'float':
        np_img_ycbcr = rgb2ycbcr(np_rgb_img) / 255
    elif type == 'int':
        np_img_ycbcr = rgb2ycbcr(np_rgb_img).astype(np.uint8)
    else:
        raise ValueError("Bad type specified")
        
    return np_img_ycbcr


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
    from skimage.feature import greycomatrix
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

def greycoprops_from_image(np_img: np.ndarray, distances: t.List[int], angles: t.List[float], prop: str):
    from skimage.feature import greycoprops
    np_comatrix = comatrix_from_image(np_img, distances, angles)
    if len(np_img.shape) == 3:
        np_result = np.empty((np_img.shape[-1], len(distances), len(angles)))
        for i in range(np_img.shape[-1]):
            np_result[i] = greycoprops(np_comatrix[:,:,i,:,:], prop=prop)
    else:
        raise ValueError ("Image shape not supported")

    return np_result


def calculate_difference_image(np_img: np.ndarray) -> np.ndarray:
    """Calculate difference image that is convolution with (1,-1) filter

    Args:
        np_img (np.ndarray): input image, should be image with integer pixels in range 0-255

    Raises:
        ValueError: len(np_img.shape) other than 2 or 3

    Returns:
        np.ndarray: calculated diffference image
    """
    #TODO compatibility with floats and ints
    from scipy.ndimage.filters import convolve
    #mask = [[-1,1,1], [-1,-2,1], [1,1,1]]
    mask = [[0,0,0], [-1,1,0], [0,0,0]]
    np_filter = np.array(mask)
    np_diff_img = np.empty(np_img.shape)
    if len(np_img.shape) == 3:
        for i in range(np_img.shape[-1]):
            np_diff_img[:, :, i] = np.abs(convolve(np_img[:, :, i], np_filter))
    elif len(np_img.shape) == 2:
        np_diff_img = np.abs(convolve(np_img, np_filter))
    else:
        raise ValueError('Bad shape of the image')
    # only positive values
    #np_diff_img[np_diff_img<0] = 0
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
