import torch
import torch.nn as nn

from toumei.objectives.misc.utils import freezeModel, unfreezeModel
from toumei.objectives.module import Module
from toumei.objectives.objective import Objective
from toumei.parameterization.generator import Generator


class Pipeline(Objective):
    """
    This is a wrapper class for objective generation in a pipeline based work-flow.
    It takes a generator and a tree like objective function definition consisting of modules, which is executed
    recursively in the forward function.
    """
    def __init__(self, generator: Generator, obj_func: Module):
        super(Pipeline, self).__init__()

        # the image generator and the objective function
        self.generator = generator
        self.obj_func = obj_func

    def attach(self, model: nn.Module):
        """
        Attach the modules to the model.
        :param model:
        :return: nothing
        """
        if self.model is not None:
            self.detach()

        self.model = model

        # freeze the model
        freezeModel(model)

        # call attach on the root
        self.root.attach(model)

    def detach(self):
        """
        Detach the modules from the current model
        :return: nothing
        """

        # check if attached
        if self.model is None:
            print("Cannot detach the current objective, since it was not detached in the first place.")
            return

        # unfreeze the model
        unfreezeModel(self.model)

        # call detach on the root
        self.root.detach()

        # reset the current model
        self.model = None

    @property
    def root(self) -> Module:
        """
        Returns the root node of the objective function tree
        :return: the root node
        """
        return self.obj_func

    def forward(self) -> torch.Tensor:
        """
        The forward function for the objective.
        This calls the root, which induces recursive calls for each child of the module
        :return: a tensor for optimization
        """
        return self.root()

