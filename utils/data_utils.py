# stdlib
import os
from random import randint, randint, uniform
from glob import glob
# 3p
from torch import unsqueeze, cat
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# This module is inspired from https://github.com/aitorzip/PyTorch-CycleGAN/blob/master/utils.py


# ----------------- Data Loader -----------------
class ImageDataset(Dataset):
    def __init__(self, root, _transforms=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(_transforms)
        self.unaligned = unaligned

        self.files_A = sorted(glob(os.path.join(root, '{}A/*.*'.format(mode))))
        self.files_B = sorted(glob(os.path.join(root, '{}B/*.*'.format(mode))))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


class DatasetLoader:
    def __init__(self, args):
        self.transforms = [transforms.Resize(int(args.size*1.12), Image.BICUBIC),
                           transforms.RandomCrop(args.size),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.args = args

    def load(self):
        args = self.args
        dataloader = DataLoader(ImageDataset(args.dataroot, _transforms=self.transforms, unaligned=True),
                                batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu)

        return dataloader


# ----------------- Replay Buffer -----------------
class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Trying to create a Anti-buffer. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if uniform(0, 1) > 0.5:
                    i = randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(cat(to_return))
