import numpy as np
import h5py
import sys
import typing as t

import tensorflow as tf
from tensorflow.keras.applications import Xception, DenseNet201
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization
from sklearn.preprocessing import MinMaxScaler

from color_space_operations import all_colorspaces_from_rgb, calculate_difference_image, rgb_to_ycbcr, rgb_to_hsv, greycoprops_from_image


def load_dataset_h5(path, dataset_name):
    with h5py.File(path, "r") as hf:
        print(hf.keys())
        X = hf[dataset_name][:]
        hf.close()
    return X


def getHCbCr(np_X, type="int"):
    for i in range(np_X.shape[0]):
        np_all = all_colorspaces_from_rgb(np_X[i], type=type)
        np_X[i,:,:,0] = np_all[:,:,3]    # H into R place
        np_X[i,:,:,1] = np_all[:,:,7]    # Cb into R place
        np_X[i,:,:,2] = np_all[:,:,8]    # Cr into R place
        del np_all
    return np_X


def getHSV(np_X, type="int"):
    for i in range(np_X.shape[0]):
        np_X[i] = rgb_to_hsv(np_X[i], type=type)
    return np_X


def getYCbCr(np_X, type="int"):
    for i in range(np_X.shape[0]):
        np_X[i] = rgb_to_ycbcr(np_X[i], type=type)
    return np_X

def getGradImg(np_X, type="int"):
    for i in range(np_X.shape[0]):
        np_X[i] = calculate_difference_image(np_X[i], kernel="grad")
    return np_X


def get_xception():
    base_model = Xception(include_top=False, weights=None, classes=2)
    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def get_densenet():
    base_model = DenseNet201(include_top=False, weights=None, classes=2)
    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def get_scalaraNN_model():
    initializer =  tf.keras.initializers.GlorotNormal()
    model = Sequential()
    model.add(Dense(18, activation='relu', kernel_initializer=initializer))
    model.add(Dense(18, activation='relu', kernel_initializer=initializer))
    model.add(Dense(18, activation='relu', kernel_initializer=initializer))
    model.add(BatchNormalization())

    model.add(Dense(10, activation='relu', kernel_initializer=initializer))
    model.add(Dense(10, activation='relu', kernel_initializer=initializer))
    model.add(Dense(10, activation='relu', kernel_initializer=initializer))
    model.add(BatchNormalization())

    model.add(Dense(5, activation='relu', kernel_initializer=initializer))
    model.add(Dense(5, activation='relu', kernel_initializer=initializer))
    model.add(Dense(5, activation='relu', kernel_initializer=initializer))

    model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

def getAdditionalScalars(np_X):
    np_X_scalars = np.empty((np_X.shape[0],18))
    for i in range(np_X.shape[0]):
        np_all = all_colorspaces_from_rgb(np_X[i], type="int")
        np_HSV = np_all[:,:,3:6]
        np_YCbCr = np_all[:,:,6:]

        np_grad_HSV = calculate_difference_image(np_HSV, kernel='grad')
        np_grad_YCbCr = calculate_difference_image(np_YCbCr, kernel='grad')

        # histogram peak_points for HSV
        np_X_scalars[i,:6] = np.array(hist_peak_point_from3D_img(np_grad_HSV))
        # histogram peak_points for YCbCr
        np_X_scalars[i,6:12] = np.array(hist_peak_point_from3D_img(np_grad_YCbCr))
        # angle 0 distance 2 for H
        np_X_scalars[i,12] =  greycoprops_from_image(np_HSV[0], distances=[2], angles=[0], prop='contrast')

        # angle 0 distance 2 for S
        np_X_scalars[i,13] =  greycoprops_from_image(np_HSV[1], distances=[2], angles=[0], prop='ASM')
        
        # angle pi/2 distance 1 for V
        np_X_scalars[i,14] =  greycoprops_from_image(np_HSV[2], distances=[1], angles=[np.pi/2], prop='contrast')
        
        # angle pi/2 distance 1 for Y
        np_X_scalars[i,15] = greycoprops_from_image(np_YCbCr[0], distances=[1], angles=[np.pi/2], prop="contrast")

        # angle pi/2 distance 2 for Cb
        np_X_scalars[i,16] = greycoprops_from_image(np_YCbCr[1], distances=[2], angles=[np.pi/2], prop="contrast")

        # angle 0 distance 2 for Cr
        np_X_scalars[i,17] = greycoprops_from_image(np_YCbCr[2], distances=[2], angles=[0], prop="contrast")


    return np_X_scalars

def scalar_from_single_img(np_X):
    np_X_scalars = np.empty(18)

    np_all = all_colorspaces_from_rgb(np_X, type="int")
    np_HSV = np_all[:,:,3:6]
    np_YCbCr = np_all[:,:,6:]

    np_grad_HSV = calculate_difference_image(np_HSV, kernel='grad')
    np_grad_YCbCr = calculate_difference_image(np_YCbCr, kernel='grad')

    # histogram peak_points for HSV
    np_X_scalars[:6] = np.array(hist_peak_point_from3D_img(np_grad_HSV))
    # histogram peak_points for YCbCr
    np_X_scalars[6:12] = np.array(hist_peak_point_from3D_img(np_grad_YCbCr))
    # angle 0 distance 2 for H
    np_X_scalars[12] =  greycoprops_from_image(np_HSV[0], distances=[2], angles=[0], prop='contrast')

    # angle 0 distance 2 for S
    np_X_scalars[13] =  greycoprops_from_image(np_HSV[1], distances=[2], angles=[0], prop='ASM')
    
    # angle pi/2 distance 1 for V
    np_X_scalars[14] =  greycoprops_from_image(np_HSV[2], distances=[1], angles=[np.pi/2], prop='contrast')
    
    # angle pi/2 distance 1 for Y
    np_X_scalars[15] = greycoprops_from_image(np_YCbCr[0], distances=[1], angles=[np.pi/2], prop="contrast")

    # angle pi/2 distance 2 for Cb
    np_X_scalars[16] = greycoprops_from_image(np_YCbCr[1], distances=[2], angles=[np.pi/2], prop="contrast")

    # angle 0 distance 2 for Cr
    np_X_scalars[17] = greycoprops_from_image(np_YCbCr[2], distances=[2], angles=[0], prop="contrast")


    return np_X_scalars

def hist_peak_point_from3D_img(np_img: np.ndarray, bins=256, hist_range=(0,255)) -> t.Tuple[float, int]:
    """Calculate histogram peek point

    Args:
        np_img (np.ndarray): input image to calculate histograms 
        bins (int, optional): number of histogram intervals. Defaults to 511.
        hist_range (tuple, optional): image pixels range. Defaults to (-255, 256).

    Returns:
        t.Tuple[float, int]: peek point coordinates (pixel value, number of occurences)
    """
    peak_points = []
    for i in range(np_img.shape[2]):
        np_hist, bins = np.histogram(np_img[i], density=True,
                                    bins=bins, range=hist_range)
        y = np_hist.max()
        idx = np.argwhere(np_hist==y)
        if len(idx)>1:
            idx = int(idx[0])
        else:
            idx = int(idx)
        
        x = int(bins[idx])
        peak_points.append(x)
        peak_points.append(y)

    return peak_points

def normalize_scalar_input(X_train, X_val):
    """Function which scales training and validation dataset usign MinMaxScaler from sk-learn

    Args:
        X_train ([np.ndarray]): [training data-set]
        X_val ([np.ndarray]): [validation data-set]

    Returns:
        t.Tuple[np.ndarray, np.ndarray]:: [scalled training and validation data-set]
    """
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    return (X_train, X_val)

