import os
import typing as t
from glob import glob
from PIL import Image

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