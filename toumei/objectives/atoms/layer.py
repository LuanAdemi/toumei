import torch

from toumei.objectives.atoms.atom import Atom
from toumei.objectives.misc.utils import convertUnitString


class Layer(Atom):
    """
    The layer objective.
    This objective optimizes the activation of a whole layer.
    """
    def __init__(self, unit: str):
        self.identifiers = convertUnitString(self.unit)

        # check if the unit string is valid
        if len(self.identifiers) != 1:
            raise Exception(f"{unit} is not a valid unit string for the layer objective.")

        super(Layer, self).__init__(unit, self.identifiers[0])

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        The forward function for the layer objective
        :param args: the arguments
        :param kwargs: the keyword arguments
        :return: the channel objective tensor
        """
        return -self.activation.mean()

    @property
    def name(self):
        return "Layer"
