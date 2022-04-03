import torch.nn as nn


class Module(object):
    def __init__(self):
        super(Module, self).__init__()

    @property
    def name(self):
        return "Module"
