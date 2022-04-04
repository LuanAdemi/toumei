import torch
import torch.nn as nn
import tqdm

from toumei.parameterization import Generator


class Objective(object):
    """
    The base class for the feature visualization objectives
    It handles the optimization process and provides a simple interface for analyzing the results
    """
    def __init__(self):
        super(Objective, self).__init__()
        self.model = None
        self.children = []

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
        print("TODO")
        return NotImplementedError

    def optimize(self, epochs=512, optimizer=torch.optim.Adam, lr=0.01):
        """
        Optimize the current objective
        :param epochs: the amount of optimization steps
        :param optimizer: the optimizer (default is Adam)
        :param lr: the learning rate (default is 0.01)
        :return: nothing
        """
        # attach the optimizer to the parameters of the current generator
        opt = optimizer(self.generator.parameters, lr)

        for e in tqdm.trange(epochs):
            opt.zero_grad()
            # forward pass using input from generator
            self.model(self.generator.getImage())

            # calculate loss using current objective function
            loss = self.forward()

            # optimize the generator
            loss.backward()
            opt.step()

    def forward(self) -> torch.Tensor:
        """
        The forward function returning the tensor calculated using the
        objective function. This needs to be overwritten by child classes
        implementing an objective.
        :return:
        """
        return NotImplementedError

    @property
    def generator(self) -> Generator:
        """
        Returns the generator object
        :return:
        """
        return NotImplementedError

