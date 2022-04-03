import torch.nn as nn


class Module(object):
    def __init__(self):
        super(Module, self).__init__()

    def __call__(self, *args, **kwargs):
        return NotImplementedError

    def attach(self, model: nn.Module):
        return NotImplementedError

    @property
    def name(self):
        return "Module"
