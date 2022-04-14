import torch.nn as nn


class GeneratorBlock(nn.Module):
    """
    A GeneratorBlock consisting out of a conv2d layer, batch norm and a LeakyRelu
    """
    def __init__(self, in_chn, out_chn, kernel_size=(4, 4), stride=(1, 1), padding=0):
        super(GeneratorBlock, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_chn, out_chn, kernel_size, stride, padding, bias=False),
            nn.ReLU()
        )

    def forward(self, x):
        return self.main(x)


class Generator(nn.Module):
    """
    The generator network architecture based on the original DCGAN paper
    """
    def __init__(self, *shape):
        super(Generator, self).__init__()

        self.first = GeneratorBlock(100, 256)
        self.block1 = GeneratorBlock(256, 128, stride=(3, 3), padding=1)
        self.block2 = GeneratorBlock(128, 64, stride=(3, 3), padding=1)
        self.block3 = GeneratorBlock(64, 32, stride=(3, 3), padding=1)

        self.last = nn.Sequential(
            nn.ConvTranspose2d(32, 3, (4, 4), (3, 3), padding=1, bias=False)
        )

    def forward(self, x):
        x = self.first(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.last(x)
