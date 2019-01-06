# stdlib
import os
from os.path import join
from glob import glob
import argparse
from shutil import copyfile
# 3p
import numpy as np
from PIL import Image


# -------------- Utils --------------
GRAY_DIR = "3gray"


def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def get_images(folder):
    "returns a list of all images in folder"
    extentions = ["png", "jpeg", "jpg"]
    images = []
    for ext in extentions:
        paths = [os.path.basename(x) for x in glob('{}/*.{}'.format(folder, ext))]
        images.extend(paths)

    return images


def r2b_to_3gray(folder):
    "read, convert all RGB imagesin folder to 3-dimentional grayscale image and save them"
    images = get_images(folder)
    create_folder(join(folder, GRAY_DIR))
    for image in images:
        im = Image.open(join(folder, image), 'r').convert('L')
        im = np.stack((im,)*3, axis=-1)
        im = Image.fromarray(im)
        im.save(join(folder, GRAY_DIR, image))


def resize_folder(folder, size=256):
    "resize all image files in folder to the desired size"
    images = get_images(folder)
    size = (size, size)
    for image in images:
        im = Image.open(join(folder, image))
        im = im.resize(size, Image.CUBIC)
        im.save(join(folder, image))


def create_dataset(folder, ds_name, size=256, train_ratio=0.9):
    """Create a dataset for training GAN models in datasets folder with name `ds_name`.
    This function will do the necessary preprocessing (resize, create BW images, create train/test datasets.
    It expects that folder has two subfolders with name `A` and `B`.
    The folder A is the one containing BW images."""

    ds_dir = join("datasets", ds_name)
    create_folder("datasets")
    create_folder(ds_dir)

    # create `B` dataset
    f_B = join(folder, "B")
    resize_folder(f_B, size=size)
    images = get_images(f_B)
    np.random.shuffle(images)
    sep = int(len(images) * train_ratio)
    train, test = images[:sep], images[sep:]
    create_folder(join(ds_dir, "trainB"))
    create_folder(join(ds_dir, "testB"))
    for name in train:
        copyfile(join(folder, "B", name), join(ds_dir, "trainB", name))
    for name in test:
        copyfile(join(folder, "B", name), join(ds_dir, "testB", name))

    # create `A` dataset
    f_A = join(folder, "A")
    resize_folder(f_A, size=size)
    r2b_to_3gray(f_A)
    images = get_images(join(f_A, GRAY_DIR))
    np.random.shuffle(images)
    sep = int(len(images) * train_ratio)
    train, test = images[:sep], images[sep:]
    create_folder(join(ds_dir, "trainA"))
    create_folder(join(ds_dir, "testA"))
    for name in train:
        copyfile(join(folder, "A", GRAY_DIR, name), join(ds_dir, "trainA", name))
    for name in test:
        copyfile(join(folder, "A", GRAY_DIR, name), join(ds_dir, "testA", name))


def create_argno_dataset(folder, ds_name, size=256, train_ratio=0.9):
    create_folder(join(folder, "temp"))
    create_folder(join(folder, "temp", "A"))
    create_folder(join(folder, "temp", "B"))
    images = get_images(folder)
    np.random.shuffle(images)
    sep = int(len(images) * 0.5)
    for i in range(sep):
        copyfile(join(folder, images[i]), join(folder, "temp", "A", images[i]))
    for i in range(sep, len(images)):
        copyfile(join(folder, images[i]), join(folder, "temp", "B", images[i]))
    create_dataset(join(folder, "temp"), ds_name, size, train_ratio)


# -------------------------------------
parser = argparse.ArgumentParser(description="Later")
parser.add_argument('-d', '--data-dir', required=False, default="datasets/test", help='root directory of the dataset')
parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment')
parser.add_argument('--type', type=int, choices=[1, 2], default=1, help='1: containing A|B, 2: contains only photos')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
args = parser.parse_args()

if __name__ == '__main__':
    if args.type == 1:
        create_dataset(args.data_dir, args.name, args.size)
    elif args.type == 2:
        create_argno_dataset(args.data_dir, args.name, args.size)
