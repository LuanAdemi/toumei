import torch

from toumei.cnns.objectives.atoms.atom import Atom
from toumei.cnns.objectives.utils import convert_unit_string


class Neuron(Atom):
    """
    The neuron objective.
    This objective optimizes the activation of a single neuron
    """
    def __init__(self, unit: str):
        """
        Initializes a new Neuron objective atom

        :param unit: the unit of the objective
        """
        self.identifiers = convert_unit_string(unit)

        # check if the unit string is valid
        if len(self.identifiers) != 3:
            raise Exception(f"{unit} is not a valid unit string for the neuron objective. Try something like: "
                            f"'model:layer:(channel:)neuron'")

        super(Neuron, self).__init__(unit, self.identifiers[0])

    @property
    def name(self):
        return "Neuron"

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        The forward function for the layer objective

        :param args: the arguments
        :param kwargs: the keyword arguments
        :return: the channel objective tensor
        """
        return -self.activation[self.identifiers[1], self.identifiers[2]].mean()

