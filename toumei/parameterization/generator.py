import torch


class Generator(object):
    def __init__(self):
        super(Generator, self).__init__()

    def getImage(self, *args, **kwargs) -> torch.Tensor:
        return NotImplementedError
