import torch

from toumei.cnns.objectives.atoms.atom import Atom
from toumei.cnns.objectives.utils import convert_unit_string


class Channel(Atom):
    """
    The channel objective.
    This objective optimizes the activation of a whole channel
    """
    def __init__(self, unit: str):
        """
        Initializes a new Channel objective atom

        :param unit: the unit of the objective
        """
        self.identifiers = convert_unit_string(unit)

        self.unit = unit
        self.layer = self.identifiers[0]

        # check if the unit string is valid
        if len(self.identifiers) != 2:
            raise Exception(f"{unit} is not a valid unit string for the layer objective.")

        # init a new atom
        super(Channel, self).__init__(unit, self.layer)

    @property
    def name(self):
        return "Channel"

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        The forward function for the channel objective

        :param args: the arguments
        :param kwargs: the keyword arguments
        :return: the channel objective tensor
        """
        return -self.activation[self.identifiers[1]].mean()
