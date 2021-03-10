import os
import typing as t
import random
from glob import glob
from PIL import Image

import numpy as np

# construct images paths
def get_files_paths_recursive(dir_path: str, extension="*.png")-> t.List[str]:
    images = [image for x in os.walk(dir_path) for image in glob(os.path.join(x[0], extension))]
    return images


def copy_files_and_resize(source_dir: str, result_dir: str, size=(256,256)):
    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)
    images = get_files_paths_recursive(SOURCE_PATH)
    i = 0
    for image_path in images:
        if i%100 == 0:
            print(f"Image number {i}")
        i += 1
        image = Image.open(image_path)
        resized_image = image.resize(SIZE)
        new_image_path = os.path.join(RESULT_PATH, os.path.basename(image_path))
        resized_image.save(new_image_path)
        
        
def get_image_data(src_path: str, random_shuffle=True)-> t.Generator[np.ndarray, None, None]:
    images_paths = get_files_paths_recursive(src_path)
    if random_shuffle:
        random.shuffle(images_paths)
    print(f"len: {len(images_paths)}")
    for image_path in images_paths[:200]:
        tmp_image = Image.open(image_path)
        np_image = np.array(tmp_image).astype(np.float32) / 255
        tmp_image.close()
        yield np_image
