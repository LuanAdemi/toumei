import torch
import torch.nn as nn
import tqdm
from toumei.cnns.featurevis.objectives.tv_loss import TVLoss
from toumei.cnns.featurevis.parameterization.imagegenerator import ImageGenerator
from toumei.general.objective import Objective


class FeatureVisualizationMethod(Objective):
    """
    The base class for the feature visualization objectives
    It handles the optimization process and provides a simple interface for analyzing the results
    """

    def __str__(self) -> str:
        return f"FeatureVisualizationMethod({self.model.__class__.__name__})"

    def attach(self, model: nn.Module):
        """
        Attach to the given model.
        :param model: The inspected model
        """
        return NotImplementedError

    def detach(self):
        """
        Detach from the current model
        """

        return NotImplementedError

    def optimize(self, epochs=512, optimizer=torch.optim.Adam, lr=5e-3, tv_loss=False, verbose=True):
        """
        Optimize the current objective
        :param verbose: Show the progress bar
        :param epochs: the amount of optimization steps
        :param optimizer: the optimizer (default is Adam)
        :param lr: the learning rate (default is 0.05)
        :param tv_loss: enable total variance loss
        """
        # send the model and the generator to the correct device
        self.model.to(self.device)
        self.model.eval()
        self.generator.to(self.device)

        # attach the optimizer to the parameters of the current generator
        opt = optimizer(self.generator.parameters, lr)

        criterion = TVLoss()

        with tqdm.trange(epochs, disable=not verbose) as t:
            t.set_description(self.__str__())
            for _ in t:
                def step():
                    # reset gradients
                    opt.zero_grad()

                    # forward pass using input from generator
                    img = self.generator.get_image().to(self.device)
                    out = self.model(img)

                    # calculate loss using current objective function
                    loss = self.forward()

                    if tv_loss:
                        loss += 0.15 * criterion(img)

                    # optimize the generator
                    loss.backward()
                    opt.step()

                    t.set_postfix(loss=loss.item())
                opt.step(step())

    def forward(self) -> torch.Tensor:
        """
        The forward function returning the tensor calculated using the
        objective function. This needs to be overwritten by child classes
        implementing an objective.
        :return:
        """
        return NotImplementedError

    def plot(self):
        """
        Plots the current feature visualization result
        """
        self.generator.plot_image()

    @property
    def generator(self) -> ImageGenerator:
        """
        Returns the generator object
        """
        return NotImplementedError


