import torch
import torch.nn as nn


def freezeModel(model: nn.Module):
    for p in model.parameters():
        p.requires_grad_(False)


def unfreezeModel(model: nn.Module):
    for p in model.parameters():
        p.requires_grad_(True)


class Objective(object):
    def __init__(self):
        super(Objective, self).__init__()
        self.model = None
        self.children = []

    def __str__(self) -> str:
        return f"Objective({self.model})"

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        This method returns the tensor for backpropagation using the objective
        :param x: the input image
        :return: the loss
        """
        _ = self.model(x)
        return self.forward()

    def attach(self, model: nn.Module):
        """
        Attach to the given model.
        :param model: The inspected model
        :return: nothing
        """

        # set the model
        self.model = model

        # freeze the model
        freezeModel(model)

        # attach each atom to the model
        for atom in self.atoms:
            atom.attach(model)

    def detach(self):
        """
        Detach from the current model
        :return: nothing
        """

        if self.model is None:
            print("[Warning] Cannot detach from current model, since objective is not attached in the first place.")
            return

        unfreezeModel(self.model)

        # remove all the hooks
        for atom in self.atoms:
            atom.detach()

    def forward(self) -> torch.Tensor:
        return 0

