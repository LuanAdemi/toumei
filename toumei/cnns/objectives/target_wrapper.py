import torch

from toumei.cnns.objectives.atoms.atom import Atom


class TargetWrapper(Atom):
    """
    Wraps an atom and injects a target activation into the atom loss

    This is useful for optimizing against specific values e.g. in a regression task.
    """

    def __init__(self, atom: Atom, target):
        """
        Initializes a new TargetWrapper object
        """
        self.atom = atom
        self.target = target

        self.loss = torch.nn.MSELoss()

        super().__init__(atom.unit, atom.layer)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        self.atom.hook_endpoint = self.hook_endpoint
        act = self.atom.forward().view(-1)
        return self.loss(act, -torch.tensor([self.target], device=torch.device(act.device), dtype=torch.float))
