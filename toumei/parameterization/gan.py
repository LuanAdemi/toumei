import torch.nn as nn
import torch
from toumei.parameterization.models.generator import Generator
from toumei.parameterization.imagegenerator import ImageGenerator


# might not be needed
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class GAN(ImageGenerator):
    def __init__(self, *shape):
        super(GAN, self).__init__()

        self.shape = shape
        self.model = Generator(*self.shape)

        # initialize the weights of the models
        self.model.apply(weights_init)

        self.fixed_noise = torch.randn(1, 100, 1, 1, requires_grad=True)

    @property
    def parameters(self) -> torch.Tensor:
        return self.model.parameters()

    def get_image(self, *args, **kwargs) -> torch.Tensor:
        return self.model(torch.randn(1, 100, 1, 1, requires_grad=True))

    @property
    def name(self) -> str:
        return "GAN"
