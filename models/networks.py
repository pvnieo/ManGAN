# stdlib

# 3P
import torch.nn as nn
import torch.nn.functional as F
# Project


# -------------------- Residual Block --------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_features, use_dropout):
        super().__init__()
        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3, padding=0),
                      nn.InstanceNorm2d(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3, padding=0),
                      nn.InstanceNorm2d(in_features)]
        if use_dropout:
            conv_block.insert(4, nn.Dropout(0.5))

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


# -------------------- Resnet Generator --------------------
class ResNetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, n_residual_blocks=9, use_dropout=False):
        super().__init__()

        # Input layer
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, 64, kernel_size=7, padding=0),
                 nn.InstanceNorm2d(64),
                 nn.ReLU(True)]

        # Encoding layers
        in_features = 64
        out_features = in_features * 2

        for _ in range(2):
            model += [nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(True)]
            in_features = out_features
            out_features = in_features * 2

        # Transformations layers (Residual blocks)
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features, use_dropout)]

        # Decoding layers
        out_features = in_features // 2

        for _ in range(2):
            model += [nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(True)]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(64, output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


# -------------------- UNet Generator --------------------
# TODO

# -------------------- Descriminator --------------------
class Descriminator(nn.Module):
    def __init__(self, input_nc):
        super(Descriminator, self).__init__()

        # Input layer
        model = [nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, True)]

        # Middle Layers
        in_features = 64
        out_features = in_features * 2

        for _ in range(2):
            model += [nn.Conv2d(in_features, out_features, kernel_size=4, stride=2, padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.LeakyReLU(0.2, True)]

            in_features = out_features
            out_features = in_features * 2

        model += [nn.Conv2d(in_features, out_features, kernel_size=4, stride=1, padding=1),
                  nn.InstanceNorm2d(out_features),
                  nn.LeakyReLU(0.2, True)]

        # Classification layer
        model += [nn.Conv2d(out_features, 1, kernel_size=4, stride=1, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
