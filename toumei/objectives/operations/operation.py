from toumei.objectives.module import Module


class Operation(Module):
    def __init__(self, *children):
        super(Operation, self).__init__()
        self.atoms = children

    def __str__(self):
        return f"{self.name}()"

    def __call__(self, *args, **kwargs):
        return self.forward(args)

    def forward(self, *args) -> int:
        return NotImplementedError

    @property
    def name(self):
        return "Operation"

    @property
    def children(self):
        return self.atoms
