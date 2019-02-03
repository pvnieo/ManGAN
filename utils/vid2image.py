import cv2
from shutil import copyfile
from glob import glob
import os
from os.path import join
from tqdm import tqdm


for i in tqdm(range(1, 7)):
    vidcap = cv2.VideoCapture('Death.Note.0{}.VF-VOSTFR.mkv'.format(i))
    success, image = vidcap.read()
    count = 0
    success = True
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*2000))    # added this line 
        success, image = vidcap.read()
        os.makedirs("/B/2_{}".format(i))
        cv2.imwrite("/B/2_{}/frame{}.jpg".format(i, count), image)     # save frame as JPEG file
        # success, image = vidcap.read()
        print('Read a new frame: \r', success, end='')
        count += 1


def get_images(folder):
    "returns a list of all images in folder"
    extentions = ["png", "jpeg", "jpg"]
    images = []
    for ext in extentions:
        paths = [os.path.basename(x) for x in glob('{}/*.{}'.format(folder, ext))]
        images.extend(paths)

    return images



# for folder in ["1", "2", "3", "4"]:
#     names = get_images(join("B", folder))
#     for name in names:
#         copyfile(join("B", folder, name), join("B", "tot", folder + name))

