# stdlib
import argparse
import sys
# project
from models.cycle_gan import ColorizationCycleGAN


class Parser:

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description='An implementation of multiple approachs to automatically colorize B/W images',
            usage='''python3 train.py <model> [<args>]

The implmented models are:
   Cycle GAN     Original paper: Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks

To show additionnal args for models: python3 train.py <model_name> --help
''')
        self.parser.add_argument(
            '--model', help='Colorization model to train', choices=["cycle_gan"])
        self.parser.add_argument(
            "-e", '--epoch', type=int, default=1, help='starting epoch')
        self.parser.add_argument(
            '--n-epochs', type=int, default=200, help='number of epochs of training')
        self.parser.add_argument(
            '-bs', '--batch-size', type=int, default=1, help='size of the batches')
        self.parser.add_argument('-d', '--dataroot', required=False,
                                 default="datasets/toto", help='root directory of the dataset')
        self.parser.add_argument(
            '--lr', type=float, default=0.0002, help='initial learning rate')
        self.parser.add_argument('--decay_epoch', type=int, default=100,
                                 help='epoch to start linearly decaying the learning rate to 0')
        self.parser.add_argument(
            '--no_cuda', action='store_true', help='Disable GPU computation')
        self.parser.add_argument(
            '--name', type=str, default='test', help='name of the experiment')
        self.parser.add_argument(
            '--size', type=int, default=256, help='size of the data crop (squared assumed)')
        self.parser.add_argument(
            '--log_freq', type=int, default=100, help='frequency of logging training results')
        self.parser.add_argument('--n_cpu', type=int, default=8,
                                 help='number of cpu threads to use during batch generation')
        self.parser.add_argument(
            '--niter', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=100,
                                 help='# of iter to linearly decay learning rate to zero')

        # Sub-parser

        # Cycle GAN additional args
        ColorizationCycleGAN.model_parser(self.parser)

    def parse_args(self):
        return self.parser.parse_args()
