import torch

from toumei.cnns.objectives.operations.operation import Operation


class Multiply(Operation):
    def __init__(self, *args):
        super(Multiply, self).__init__()
        self.atoms = args

    def forward(self, *args) -> torch.Tensor:
        result = self.atoms[0]()
        for atom in self.atoms[0:]:
            result = torch.mul(result, atom())

        return result
