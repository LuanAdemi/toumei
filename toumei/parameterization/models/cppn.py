from collections import OrderedDict

import torch
import torch.nn as nn

import numpy as np


def weights_init(module):
    if isinstance(module, torch.nn.Conv2d):
        torch.nn.init.normal_(module.weight, 0, np.sqrt(1 / module.in_channels))
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)


class CompositeActivation(torch.nn.Module):
    def forward(self, x):
        x = torch.atan(x)
        return torch.cat([x/0.67, (x*x)/0.6], 1)


class CPPN(nn.Module):
    def __init__(self, num_blocks):
        super(CPPN, self).__init__()

        self.blocks = []

        self.blocks.append((f"first", nn.Conv2d(2, 16, kernel_size=1)))
        self.blocks.append((f"activation_{0}", CompositeActivation()))

        for i in range(num_blocks - 2):
            self.blocks.append((f"conv_{i + 1}", nn.Conv2d(32, 16, kernel_size=1)))
            self.blocks.append((f"activation_{i + 1}", CompositeActivation()))

        self.blocks.append((f"last", nn.Conv2d(32, 3, kernel_size=1)))
        self.blocks.append((f"out", nn.Sigmoid()))

        self.model = nn.Sequential(OrderedDict(self.blocks))

        # initialize the weights by sampling from a normal distribution
        self.model.apply(weights_init)
        # set the last layers weights to zero
        torch.nn.init.zeros_(dict(self.model.named_children())["last"].weight)

    def forward(self, x):
        return self.model(x)
