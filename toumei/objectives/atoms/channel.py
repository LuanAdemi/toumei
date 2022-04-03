from toumei.objectives.atoms.atom import Atom
from toumei.objectives.misc.utils import convertUnitString


class Channel(Atom):
    def __init__(self, unit: str):
        self.identifiers = convertUnitString(unit)

        if len(self.identifiers) != 2:
            raise Exception(f"{unit} is not a valid unit string for the layer objective.")

        super(Channel, self).__init__(unit, self.identifiers[0])

    @property
    def name(self):
        return "Channel"

    def forward(self, *args, **kwargs):
        return -self.activation[self.identifiers[1]].mean()
