from toumei.objectives.atom import Atom
from toumei.objectives.utils import convertUnitString


class Channel(Atom):
    def __init__(self, unit: str):
        super(Channel, self).__init__()
        self.unit = unit
        self.identifiers = convertUnitString(self.unit)

        if len(self.identifiers) != 2:
            raise Exception(f"{self.unit} is not a valid unit string for the layer objective.")

    def __str__(self):
        return f"Channel({self.unit})"

    def __call__(self, activation):
        return -activation[self.layer][self.identifiers[1]].mean()

    @property
    def key(self):
        return self.unit

    @property
    def layer(self):
        return self.identifiers[0]
