from typing import Iterator

import torch
from torch.nn import Parameter

from toumei.cnns.parameterization import ImageGenerator
from toumei.models import CPPN


class Neural(ImageGenerator):
    """
    An implementation of CPPN's for feature visualization
    """
    def __init__(self, *shape):
        """
        Initializes a new CPPN image generator

        :param shape: the shape of the image
        """
        super(Neural, self).__init__()

        self.shape = shape
        self.device = torch.device("cpu")
        self.cppn = CPPN(8)

        # build the input plane using polar coordinates
        r = 3 ** 0.5
        coord_range = torch.linspace(-r, r, self.shape[-1])
        # create a polar mesh grid
        x, y = torch.meshgrid(coord_range, coord_range, indexing='ij')
        # the input tensor
        self.input_tensor = torch.stack([x, y], dim=0).unsqueeze(0)

    def get_image(self, *args, **kwargs) -> torch.Tensor:
        return self.cppn(self.input_tensor)

    def to(self, device: torch.device):
        self.device = device
        self.cppn.to(device)
        self.input_tensor = self.input_tensor.to(device)

    @property
    def parameters(self) -> Iterator[Parameter]:
        return self.cppn.parameters()

    @property
    def name(self) -> str:
        return "Neural"
