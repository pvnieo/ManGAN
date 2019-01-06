# stdlib
import csv
import datetime
import os
# 3p
import numpy as np
from visdom import Visdom
from pprint import pprint
import torch
from PIL import Image
# project
from .utils import create_dir_in_checkpoint, CHECKPOINT_DIR
from .html import HTML


class Logger:
    def __init__(self, args):
        self.vis = Visdom()
        self.csv_path = os.path.join(CHECKPOINT_DIR, args.name) + '/{}.csv'.format(str(datetime.datetime.now()))
        self.is_header_written = False
        self.name = args.name
        self.web_dir = os.path.join(CHECKPOINT_DIR, args.name, 'web')
        create_dir_in_checkpoint(os.path.join(args.name, 'web'))
        self.img_dir = os.path.join(self.web_dir, 'images')
        create_dir_in_checkpoint(os.path.join(args.name, 'web', 'images'))

    def _write_row(self, row):
        with open(self.csv_path, "a") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def tensor2image(self, input_image, imtype=np.uint8):
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()
        if image_numpy.shape[0] == 1:
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        return image_numpy.astype(imtype)

    def log_iter(self, epoch, iter_, tot_iter, losses, images):
        # write header
        if not self.is_header_written:
            header = ["epoch", "iter"] + list(losses.keys())
            self._write_row(header)
            self.is_header_written = True

        # write row
        self._write_row([epoch, iter_] + list(losses.values()))

        # Plot current losses in Visdom
        self.plot_current_losses(losses, epoch, iter_, tot_iter)

        # Display current result
        self.display_current_results(images, epoch)

    def plot_current_losses(self, losses, epoch, iter_, tot_iter):
        "Inspired from: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/util/visualizer.py"
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(epoch + (iter_ / tot_iter))
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
                Y=np.array(self.plot_data['Y']),
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=1)
        except Exception as e:
            print('\n\nCould not connect to Visdom server')
            print(e)
            exit(1)

    def save_image(self, image_numpy, image_path):
        image_pil = Image.fromarray(image_numpy)
        image_pil.save(image_path)

    def display_current_results(self, visuals, epoch):
        "Inspired from: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/util/visualizer.py"
        ncols = 4
        h, w = next(iter(visuals.values())).shape[:2]
        table_css = """<style>
                table {border-collapse: separate; border-spacing:4px; white-space:nowrap; text-align:center}
                table td {width: %dpx; height: %dpx; padding: 4px; outline: 4px solid black}
                </style>""" % (w, h)
        title = self.name
        label_html = ''
        label_html_row = ''
        images = []
        idx = 0
        for label, image in visuals.items():
            image_numpy = self.tensor2image(image)
            label_html_row += '<td>%s</td>' % label
            images.append(image_numpy.transpose([2, 0, 1]))
            idx += 1
            if idx % ncols == 0:
                label_html += '<tr>%s</tr>' % label_html_row
                label_html_row = ''
        white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
        while idx % ncols != 0:
            images.append(white_image)
            label_html_row += '<td></td>'
            idx += 1
        if label_html_row != '':
            label_html += '<tr>%s</tr>' % label_html_row
        # pane col = image row
        try:
            self.vis.images(images, nrow=ncols, win=1 + 1,
                            padding=2, opts=dict(title=title + ' images'))
            label_html = '<table>%s</table>' % label_html
            self.vis.text(table_css + label_html, win=1 + 2,
                          opts=dict(title=title + ' labels'))
        except Exception as e:
            print('\n\nCould not connect to Visdom server')
            print(e)
            exit(1)

        # save images to a html file
        for label, image in visuals.items():
            image_numpy = self.tensor2image(image)
            img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
            self.save_image(image_numpy, img_path)
        # update website
        webpage = HTML(self.web_dir, 'Experiment name = %s' % self.name, reflesh=1)
        for n in range(epoch, 0, -1):
            webpage.add_header('epoch [%d]' % n)
            ims, txts, links = [], [], []

            for label, image_numpy in visuals.items():
                image_numpy = self.tensor2image(image)
                img_path = 'epoch%.3d_%s.png' % (n, label)
                ims.append(img_path)
                txts.append(label)
                links.append(img_path)
            webpage.add_images(ims, txts, links, width=256)
        webpage.save()

    def log_epoch(self, duration, losses):
        print("Took {}s".format(duration))
        pprint(losses)
        print("-------------------------")
