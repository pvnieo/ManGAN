import os
from os.path import join
from glob import glob
import argparse
from shutil import copyfile
# 3p
import numpy as np
from PIL import Image




def get_images(folder):
    "returns a list of all images in folder"
    extentions = ["png", "jpeg", "jpg"]
    images = []
    for ext in extentions:
        paths = [os.path.basename(x) for x in glob('{}/*.{}'.format(folder, ext))]
        images.extend(paths)

    return images
folder = "ONE"

images = get_images("one/")
np.random.shuffle(images)
n = int(len(images) * 0.8)
for name in images[:n]:
    copyfile(join("one", name), join("colone", "train", name))
for name in images[n:]:
    copyfile(join("one", name), join("colone", "val", name))
"""
names = get_images(folder)
for name in names:
    AB_path = "ONE/" + name
    Image.open(AB_path)
    AB = Image.open(AB_path)
    w, h = AB.size
    w3 = int(w / 3)
    h3 = int(h / 3)
    # A = AB.crop((0, 0, w3, h))
    # B = AB.crop((w3, 0, 2 * w3, h))
    # C = AB.crop((2 * w3, 0, w, h))
    A = AB.crop((0, 0, w, h3))
    B = AB.crop((0, h3, w, 2 * h3))
    C = AB.crop((0, 2 * h3, w, h))
    A.save('one/1_' + name)
    B.save('one/2_' + name)
    C.save('one/3_' + name)
"""


