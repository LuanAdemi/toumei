from typing import Iterator

import torch.nn as nn
import torch
from torch.nn import Parameter

from toumei.misc.models.generator import Generator
from toumei.parameterization.imagegenerator import ImageGenerator


# might not be needed
# custom weights initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class GAN(ImageGenerator):
    """
    A WIP implementation of GANs for feature visualization.
    """
    def __init__(self, *shape):
        """
        Initializes a new GAN image generator

        :param shape: the shape of the image
        """
        super(GAN, self).__init__()

        self.shape = shape
        self.gan = Generator()

        # initialize the weights of the models
        self.gan.apply(weights_init)

    def get_image(self, *args, **kwargs) -> torch.Tensor:
        return self.gan(torch.randn(1, 100, 1, 1, requires_grad=True))

    @property
    def parameters(self) -> Iterator[Parameter]:
        return self.gan.parameters()

    @property
    def name(self) -> str:
        return "GAN"
