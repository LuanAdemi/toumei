import torch
import torch.nn as nn
import tqdm
from toumei.parameterization.models.tv_loss import TVLoss

import toumei.parameterization
from toumei.parameterization import ImageGenerator


class Objective(object):
    """
    The base class for the feature visualization objectives
    It handles the optimization process and provides a simple interface for analyzing the results
    """
    def __init__(self):
        super(Objective, self).__init__()
        self.model = None
        self.children = []
        self.device = torch.device("cpu")

    def __str__(self) -> str:
        return f"Objective({self.model.__class__.__name__})"

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

    def summary(self):
        """
        Prints an overview of the current objective
        :return: nothing
        """
        print(f"Objective(")
        print(f"    Generator:  {self.generator}")
        print(f"    Criterion:  ")
        print(")")

    def total_variation(self, y):
        return torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + \
               torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))

    def optimize(self, epochs=512, optimizer=torch.optim.Adam, lr=5e-2):
        """
        Optimize the current objective
        :param epochs: the amount of optimization steps
        :param optimizer: the optimizer (default is Adam)
        :param lr: the learning rate (default is 0.05)
        :return: nothing
        """
        # send the model to the correct device
        self.model.to(self.device)

        # attach the optimizer to the parameters of the current generator
        opt = optimizer(self.generator.parameters, lr)

        criterion = TVLoss()

        for _ in tqdm.trange(epochs):
            # reset gradients
            opt.zero_grad()

            # forward pass using input from generator
            img = self.generator.get_image().to(self.device)
            self.model(img)

            # calculate loss using current objective function
            loss = self.forward() + 0.25 * criterion(img)

            # optimize the generator
            loss.backward()
            opt.step()

    def to(self, device: torch.device):
        """
        Sets the device for the optimization process
        :param device: the device
        :return: nothing
        """
        self.device = device

    def forward(self) -> torch.Tensor:
        """
        The forward function returning the tensor calculated using the
        objective function. This needs to be overwritten by child classes
        implementing an objective.
        :return:
        """
        return NotImplementedError

    @property
    def generator(self) -> ImageGenerator:
        """
        Returns the generator object
        :return:
        """
        return NotImplementedError


