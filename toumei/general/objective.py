import torch
import torch.nn as nn


class Objective(object):
    """
    The base class for the feature visualization objectives
    It handles the optimization process and provides a simple interface for analyzing the results
    """
    def __init__(self):
        super(Objective, self).__init__()
        self.model = None
        self.device = torch.device("cpu")

    def __str__(self) -> str:
        return f"Objective({self.model.__class__.__name__})"

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

    def optimize(self, *args, **kwargs):
        """
        Optimize the current objective

        :param args: The arguments
        :param kwargs: The keyword arguments
        """

        return NotImplementedError

    def to(self, device: torch.device):
        """
        Sets the device for the optimization process
        :param device: the device
        """
        self.device = device


