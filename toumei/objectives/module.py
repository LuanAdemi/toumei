import torch.nn as nn


class Module(object):
    """
    The base module for objective building.
    This acts as a parent class for objective atoms and objective operations
    """
    def __init__(self):
        super(Module, self).__init__()

    def __call__(self, *args, **kwargs):
        """
        The call method for the module
        :param args: arguments
        :param kwargs: keyword arguments
        :return: nothing
        """
        return NotImplementedError

    def attach(self, model: nn.Module):
        """
        Attach the module to the specified model
        :param model: the model to attach to
        :return: nothing
        """
        return NotImplementedError

    def detach(self):
        """
        Detach the module from the current model
        :return: nothing
        """
        return NotImplementedError

    @property
    def name(self):
        """
        Returns the name of the module
        :return: the name
        """
        return "Module"
