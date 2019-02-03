# stdlib
from itertools import chain
import argparse
import os
# 3P
import torch
import torch.nn as nn
from torch import save
from torch.cuda import FloatTensor
from torch.autograd import Variable
# Project
from utils.utils import CHECKPOINT_DIR
from utils.utils import lambda_scheduler, init_weights_normal
from utils.data_utils import ReplayBuffer
from .networks import ResNetGenerator, Descriminator


class ColorizationCycleGAN(nn.Module):
    @staticmethod
    def model_parser(parser):
        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')

    @property
    def name(self):
        return "colorization_cycle_gan"

    def __init__(self, args):
        super().__init__()
        self.input_nc = 3
        self.output_nc = 3

        # Path to save models
        self.path = os.path.join(CHECKPOINT_DIR, args.name)

        # Define networks
        self.G_A2B = ResNetGenerator(input_nc=self.input_nc, output_nc=self.output_nc, n_residual_blocks=9, use_dropout=False)
        self.G_B2A = ResNetGenerator(input_nc=self.output_nc, output_nc=self.input_nc, n_residual_blocks=9, use_dropout=False)
        self.D_A2B = Descriminator(self.output_nc)
        self.D_B2A = Descriminator(self.input_nc)

        self.net_names = ["G_A2B", "G_B2A", "D_A2B", "D_B2A"]

        if not args.no_cuda:
            self.G_A2B.cuda()
            self.G_B2A.cuda()
            self.D_A2B.cuda()
            self.D_B2A.cuda()

        # Define Losses
        self.criterion_gan = nn.MSELoss().cuda()
        self.criterion_cycle = nn.L1Loss().cuda()
        self.criterion_idt = nn.L1Loss().cuda()

        # Define optimizers
        self.optimizer_G = torch.optim.Adam(chain(self.G_A2B.parameters(),
                                                  self.G_B2A.parameters()), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(chain(self.D_A2B.parameters(),
                                                  self.D_B2A.parameters()), lr=0.0002, betas=(0.5, 0.999))

        # Define LR schedulers
        self.lr_scheduler_G = lambda_scheduler(self.optimizer_G, args)
        self.lr_scheduler_D = lambda_scheduler(self.optimizer_D, args)

        # Define fake buffers
        self.A2B_fake_buffer = ReplayBuffer()
        self.B2A_fake_buffer = ReplayBuffer()

        # Inputs & targets memory allocation
        Tensor = FloatTensor if not args.no_cuda else torch.Tensor
        self.input_A = Tensor(args.batch_size, args.input_nc, args.size, args.size).cuda()  # To be changed
        self.input_B = Tensor(args.batch_size, args.output_nc, args.size, args.size).cuda()  # To be changed
        self.target_real = Variable(Tensor([[args.batch_size]]).fill_(1.0), requires_grad=False).cuda()  # To be changed
        self.target_fake = Variable(Tensor([[args.batch_size]]).fill_(0.0), requires_grad=False).cuda()  # To be changed

        # Init networks
        # if args.train:
        if True:
            self.G_A2B.apply(init_weights_normal)
            self.G_B2A.apply(init_weights_normal)
            self.D_A2B.apply(init_weights_normal)
            self.D_B2A.apply(init_weights_normal)
        # elif args.test:
            # pass

    def update_lr(self):
        self.lr_scheduler_G.step()
        self.lr_scheduler_D.step()

    def forward(self):
        # GAN loss
        self.fake_A = self.G_B2A(self.real_B)
        self.fake_B = self.G_A2B(self.real_A)

        # Cycle loss
        self.rec_B = self.G_A2B(self.fake_A)
        self.rec_A = self.G_B2A(self.fake_B)

        # Identity loss
        self.same_A = self.G_B2A(self.real_A)
        self.same_B = self.G_A2B(self.real_B)

        # Images to log
        self.images = {"real_A": self.real_A, "real_B": self.real_B, "fake_A": self.fake_A, "fake_B": self.fake_B,
                       "rec_A": self.rec_A, "rec_B": self.rec_B, "same_A": self.same_A, "same_B": self.same_B}

    def backward_G(self):
        # Identity loss
        loss_id_A = self.criterion_idt(self.same_A, self.real_A) * 5
        loss_id_B = self.criterion_idt(self.same_B, self.real_B) * 5

        # GAN loss
        loss_gan_A2B = self.criterion_gan(self.D_A2B(self.fake_B), self.target_real)
        loss_gan_B2A = self.criterion_gan(self.D_B2A(self.fake_A), self.target_real)

        # Cycle loss
        loss_cycle_A = self.criterion_cycle(self.real_A, self.rec_A) * 10
        loss_cycle_B = self.criterion_cycle(self.real_B, self.fake_B) * 10

        # Update
        loss = loss_id_A + loss_id_B + loss_gan_A2B + \
            loss_gan_B2A + loss_cycle_A + loss_cycle_B
        loss.backward()

        self.losses = {"loss_id_A": loss_id_A, "loss_id_B": loss_id_B, "loss_gan_A2B": loss_gan_A2B,
                       "loss_gan_B2A": loss_gan_B2A, "loss_cycle_A": loss_cycle_A, "loss_cycle_B": loss_cycle_B}

    def backward_D(self):
        # Descriminator A
        pred_real = self.D_A2B(self.real_B)
        loss = self.criterion_gan(pred_real, self.target_real) * 0.5

        fake_B = self.A2B_fake_buffer.push_and_pop(self.fake_B)
        pred_fake = self.D_A2B(fake_B.detach())
        loss += self.criterion_gan(pred_fake, self.target_fake) * 0.5

        loss.backward()
        self.losses["loss_D_A2B"] = loss

        # Descriminator B
        pred_real = self.D_B2A(self.real_A)
        loss = self.criterion_gan(pred_real, self.target_real) * 0.5

        fake_A = self.B2A_fake_buffer.push_and_pop(self.fake_A)
        pred_fake = self.D_B2A(fake_A.detach())
        loss += self.criterion_gan(pred_fake, self.target_fake) * 0.5

        loss.backward()
        self.losses["loss_D_B2A"] = loss

    def fit(self, batch):
        # Set model input
        self.real_A = Variable(self.input_A.copy_(batch['A'])).cuda()
        self.real_B = Variable(self.input_B.copy_(batch['B'])).cuda()

        # forward pass
        self.forward()

        # Backpropagate G
        self.set_requires_grad([self.D_A2B, self.D_B2A], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        self.set_requires_grad([self.D_A2B, self.D_B2A], True)

        # Backpropagate D
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # Return images and losses
        return self.losses, self.images

    def save_model(self, epoch):
        path = self.path + "/e_" + str(epoch) + "_{}.pth"
        path_latest = self.path + "/latest_{}.pth"
        for name in self.net_names:
            net = getattr(self, name)
            save(net.state_dict(), path.format(name))
            save(net.state_dict(), path_latest.format(name))

    def set_requires_grad(self, nets, requires_grad=False):
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
