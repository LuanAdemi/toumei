import torch

from toumei.parameterization import Generator


class GANImage(Generator):
    def __init__(self):
        super(GANImage, self).__init__()

    @property
    def parameters(self) -> list:
        return NotImplementedError

    def get_image(self, *args, **kwargs) -> torch.Tensor:
         return NotImplementedError