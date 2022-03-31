import torch.nn as nn


class Objective(object):
    def __init__(self):
        super(Objective, self).__init__()
        self.activation = []
        self.model = nn.Module()

    def __str__(self):
        return f"Objective({self.model})"

    def __call__(self, x):
        _ = self.model(x)
        return self.forward(self.activation)

    def attach(self, model: nn.Module, layer: str, name: str):
        self.model = model

        def create_hook(name):
            def hook(m, i, o):
                # copy the output of the given layer
                self.activation[name] = o.unsqueeze(0)

            return hook

        self.model[layer].register_forward_hook(create_hook(name))

    def forward(self, activation):
        return NotImplementedError

