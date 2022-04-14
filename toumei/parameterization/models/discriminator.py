import torch.nn as nn


class DiscriminatorBlock(nn.Module):
    """
    A DiscriminatorBlock consisting out of a conv2d layer, batch norm and a LeakyRelu
    """
    def __init__(self, in_chn, out_chn):
        super(DiscriminatorBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_chn, out_chn, (4, 4), (2, 2), 1, bias=False),
            nn.BatchNorm2d(out_chn),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    """
    The discriminator network architecture based on the original DCGAN paper
    """
    def __init__(self, w, h, ch):
        super(Discriminator, self).__init__()
        self.first = nn.Sequential(
            nn.Conv2d(ch, 64, (4, 4), (2, 2), 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True))

        self.block1 = DiscriminatorBlock(64, 128)
        self.block2 = DiscriminatorBlock(128, 256)
        self.block3 = DiscriminatorBlock(256, 512)

        self.last = nn.Sequential(
            nn.Conv2d(512, 1, (4, 4), (2, 2), 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.first(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.last(x)
