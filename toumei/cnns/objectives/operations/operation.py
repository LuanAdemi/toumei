import torch

import torch.nn as nn

from toumei.cnns.objectives.atoms import Atom


class Operation(Atom):
    def __init__(self, *children: Atom):
        super(Operation, self).__init__("Operation", "")
        self.atoms = children

    def __str__(self):
        return f"{self.name}()"

    def __call__(self, *args, **kwargs):
        return self.forward(args)

    def attach(self, model: nn.Module):
        for child in self.children:
            child.attach(model)

    def detach(self):
        for child in self.children:
            child.detach()

    def forward(self, *args) -> torch.Tensor:
        return NotImplementedError

    @property
    def name(self):
        return "Operation"

    @property
    def children(self):
        return self.atoms
