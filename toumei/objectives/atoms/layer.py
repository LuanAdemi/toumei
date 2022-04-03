from toumei.objectives.atoms.atom import Atom
from toumei.objectives.misc.utils import convertUnitString


class Layer(Atom):
    def __init__(self, unit: str):
        self.identifiers = convertUnitString(self.unit)

        if len(self.identifiers) != 1:
            raise Exception(f"{unit} is not a valid unit string for the layer objective.")

        super(Layer, self).__init__(unit, self.identifiers[0])

    def forward(self, *args, **kwargs):
        return -self.activation.mean()

    @property
    def name(self):
        return "Layer"
