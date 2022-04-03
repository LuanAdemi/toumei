import torch


class Generator(object):
    def __init__(self):
        super(Generator, self).__init__()

    def getImage(self, z: int = None) -> torch.Tensor:
        return torch.rand(2)
