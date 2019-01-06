# stdlib
import os
# 3P
from torch.nn import init
from torch.optim.lr_scheduler import LambdaLR


# --------------- Create checkpoint directory ---------------
CHECKPOINT_DIR = 'checkpoints'
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)


def create_dir_in_checkpoint(name):
    path = os.path.join(CHECKPOINT_DIR, name)
    if not os.path.exists(path):
        os.makedirs(path)


# --------------- Lambda lr scheduler ---------------
def lambda_scheduler(optimizer, args):
    def lambda_rule(epoch):
        return 1.0 - max(0, epoch + args.epoch - args.niter) / float(args.niter_decay + 1)

    return LambdaLR(optimizer, lambda_rule)


# --------------- Init weights ---------------
def init_weights_normal(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        init.normal(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(model.weight.data, 1.0, 0.02)
        init.constant(model.bias.data, 0.0)

