from toumei.objectives.objective import Objective

import torch.nn as nn


class Sequential(Objective):
    def __init__(self, *atoms):
        super(Sequential, self).__init__()
        self.atoms = []

        for atom in atoms:
            self.atoms.append(atom)

    def attach(self, model: nn.Module, **kwargs):
        for atom in self.atoms:
            super(Sequential, self).attach(model, atom.layer, atom.key)

