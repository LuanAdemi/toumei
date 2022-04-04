import torch
import torch.nn as nn


class Objective(object):
    def __init__(self):
        super(Objective, self).__init__()
        self.model = None
        self.children = []

    def __str__(self) -> str:
        return f"Objective({self.model.__class__.__name__})"

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
        return NotImplementedError

    def detach(self):
        """
        Detach from the current model
        :return: nothing
        """

        return NotImplementedError

    def optimize(self, epochs=512, optimizer=torch.optim.Adam, lr=0.01):
        opt = optimizer([self.forward()], lr=lr)
        for e in range(epochs):
            opt.zero_grad()


    def forward(self) -> torch.Tensor:
        return NotImplementedError

