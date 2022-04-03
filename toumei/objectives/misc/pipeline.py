import torch

from toumei.objectives.objective import Objective
from toumei.objectives.atoms.atom import Atom

import torch.nn as nn

from toumei.objectives.operations.operation import Operation
from toumei.parameterization.generator import Generator


class Pipeline(Objective):
    def __init__(self, *elements):
        super(Pipeline, self).__init__()
        self.elements = elements

        if len(elements) < 2:
            raise Exception("Expected at least two elements in the pipeline (an image generator and an atom)")

        if not isinstance(self.elements[0], Generator):
            raise Exception("First element has to be an image generator.")

    def attach(self, model: nn.Module, **kwargs):
        for e in self.elements:
            if isinstance(e, Atom):
                super(Pipeline, self).attach(model, e.module, e.key)
            if isinstance(e, Operation):
                self.attach(model)

    def forward(self, activation) -> torch.Tensor:
        return 0

