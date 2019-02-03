# stdlib
import os
from os.path import join
from glob import glob
import argparse
from shutil import copyfile
from tqdm import tqdm
# 3p
import numpy as np
from PIL import Image
import cv2


def get_panels(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cropped = []
    for i, con in enumerate(contours):
        con = con.reshape(-1, 2)
        std = np.max(con, axis=0) - np.min(con, axis=0)
        if std[0] > 100 and std[1] > 100:
            min_ = np.min(con, axis=0)
            max_ = np.max(con, axis=0)
            crop_img = img[min_[1]:max_[1], min_[0]:max_[0]]
            cropped.append(crop_img)
    return cropped


def get_images(folder):
    "returns a list of all images in folder"
    extentions = ["png", "jpeg", "jpg"]
    images = []
    for ext in extentions:
        paths = [os.path.basename(x) for x in glob('{}/*.{}'.format(folder, ext))]
        images.extend(paths)

    return images

folder = 'tot'
output = 'cropped'
images = get_images(folder)

for image in tqdm(images):
    img = cv2.imread(join(folder, image))
    pannels = get_panels(img)
    for i, pannel in enumerate(pannels):
        cv2.imwrite(join(output, "{}_{}".format(i, image)), pannel)
