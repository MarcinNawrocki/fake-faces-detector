import numpy as np
import h5py
import sys

from tensorflow.keras.applications import Xception, DenseNet201
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from utils.color_space_operations import all_colorspaces_from_rgb, rgb_to_ycbcr, rgb_to_hsv, calculate_difference_image


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