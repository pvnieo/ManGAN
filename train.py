# stdlib
import sys
import argparse
from time import time
from math import ceil
# 3p
import torch
from tqdm import tqdm
# project
from utils.utils import create_dir_in_checkpoint
from utils.data_utils import DatasetLoader
from utils.logger import Logger
from utils.parser import Parser
from models.cycle_gan import ColorizationCycleGAN


parser = Parser()
args = parser.parse_args()


# Warning
if torch.cuda.is_available() and args.no_cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda\nDo you want to continue without using GPU?")
    response = str(input("  [YES to continue | ANY key to exit and restart the execution]: "))
    if response != "YES":
        sys.exit()

# Create Checkpoints dir
create_dir_in_checkpoint(args.name)


if __name__ == '__main__':
    # Load dataset
    data_loader = DatasetLoader(args).load()

    # Init Logger
    logger = Logger(args)

    # Init model
    if args.model == "cycle_gan":
        model = ColorizationCycleGAN(args)
    else:
        sys.exit()

    # Training
    for epoch in range(args.epoch, args.n_epochs):
        since = time()
        tot_iter = ceil(len(data_loader.dataset)) / args.batch_size
        for i, batch in tqdm(enumerate(data_loader), desc="Epoch {} / {}".format(epoch, args.n_epochs)):
            losses, images = model.fit(batch)
            losses = {loss: value.item() for loss, value in losses.items()}

            # Log iteration
            if (i + 1) % args.log_freq == 0:
                logger.log_iter(epoch, i, tot_iter, losses, images)

        # Update learning rates
        model.update_lr()

        # Log epoch
        duration = int(time() - since)
        logger.log_epoch(duration, losses)

        # Save model
        if epoch % 5 == 0:
            model.save_model(epoch)

