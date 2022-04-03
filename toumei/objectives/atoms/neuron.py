from toumei.objectives.atoms.atom import Atom
from toumei.objectives.misc.utils import convertUnitString


class Neuron(Atom):
    def __init__(self, unit: str):
        self.identifiers = convertUnitString(unit)

        if len(self.identifiers) != 3:
            raise Exception(f"{unit} is not a valid unit string for the neuron objective. Try something like: "
                            f"'model:layer:(channel:)neuron'")

        super(Neuron, self).__init__(unit, self.identifiers[0])

    @property
    def name(self):
        return "Neuron"

    def forward(self, *args, **kwargs):
        return -self.activation[self.identifiers[1], self.identifiers[2]].mean()

