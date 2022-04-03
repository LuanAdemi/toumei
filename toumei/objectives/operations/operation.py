from toumei.objectives.module import Module
import torch.nn as nn


class Operation(Module):
    def __init__(self, *children):
        super(Operation, self).__init__()
        self.atoms = children

    def __str__(self):
        return f"{self.name}()"

    def __call__(self, *args, **kwargs):
        return self.forward(args)

    def attach(self, model: nn.Module):
        for child in self.children:
            child.attach(model)

    def forward(self, *args) -> int:
        return NotImplementedError

    @property
    def name(self):
        return "Operation"

    @property
    def children(self):
        return self.atoms
