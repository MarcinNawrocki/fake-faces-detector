import os
import typing as t
import random
from glob import glob
from PIL import Image

import numpy as np

# construct images paths


def get_files_paths_recursive(dir_path: str, extension="*.png") -> t.List[str]:
    images = [image for x in os.walk(dir_path)
              for image in glob(os.path.join(x[0], extension))]
    return images


def copy_files_and_resize(source_dir: str, result_dir: str, size=(256, 256)):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    images = get_files_paths_recursive(source_dir)
    i = 0
    for image_path in images:
        if i % 100 == 0:
            print(f"Image number {i}")
        i += 1
        image = Image.open(image_path)
        resized_image = image.resize(size)
        new_image_path = os.path.join(result_dir, os.path.basename(image_path))
        resized_image.save(new_image_path)


def get_image_data(src_path: str, random_shuffle=True, max_number_of_images=1000, type='float', grayscale=False) -> t.Generator[np.ndarray, None, None]:
    images_paths = get_files_paths_recursive(src_path)
    if random_shuffle:
        random.shuffle(images_paths)
    images_paths = images_paths[:max_number_of_images]
    print(f"len: {len(images_paths)}")
    for image_path in images_paths:
        tmp_image = Image.open(image_path)
        if grayscale:
            tmp_image = tmp_image.convert('L')
            
        if type == 'float':
            np_image = np.array(tmp_image).astype(np.float32) / 255
        elif type == 'int':
            np_image = np.array(tmp_image).astype(np.uint8)
        else:
            raise ValueError('Bad data type specified')
        tmp_image.close()
        yield np_image

def construct_db_from_dirs(source_dirs: t.List[str], result_dir: str, extension=".png"):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    images = []
    for source_dir in source_dirs:
        tmp_imgs = []
        tmp_imgs = get_files_paths_recursive(source_dir)
        images += tmp_imgs
    i=0
    for image_path in images:
        if i%100 == 0:
            print(f"Image number {i}")
        i += 1
        image = Image.open(image_path)
        new_image_path = os.path.join(result_dir, (str(i)+extension))
        image.save(new_image_path)
