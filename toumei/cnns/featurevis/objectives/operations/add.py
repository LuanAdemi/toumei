import torch

from toumei.cnns.featurevis.objectives.operations.operation import Operation


class Add(Operation):
    def __init__(self, *args):
        super(Add, self).__init__()
        self.atoms = args

    def forward(self, *args) -> torch.Tensor:
        result = self.atoms[0]()
        for atom in self.atoms[0:]:
            result = torch.add(result, atom())

        return result
