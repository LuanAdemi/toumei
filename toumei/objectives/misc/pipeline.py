import torch
import torch.nn as nn

from toumei.objectives.module import Module
from toumei.objectives.objective import Objective, freezeModel
from toumei.parameterization.generator import Generator


class Pipeline(Objective):
    def __init__(self, generator: Generator, obj_func: Module):
        super(Pipeline, self).__init__()
        self.generator = generator
        self.obj_func = obj_func

    def attach(self, model: nn.Module):
        self.model = model
        freezeModel(model)
        self.root.attach(model)

    @property
    def root(self) -> Module:
        return self.obj_func

    def forward(self) -> torch.Tensor:
        return self.root()

