import torch.nn as nn


def add_layers(model, dimensions):
    layers = []
    for i, (dimension) in enumerate(dimensions):
        fc = nn.Linear(dimension, dimensions[i + 1])
        setattr(model, f"fc{i}", fc)
        layers.append(fc)
        if len(dimensions) == i + 2:
            break

    return layers


class SimpleMLP(nn.Module):
    layers = []

    def __init__(self, *dimensions):
        super(SimpleMLP, self).__init__()

        self.layers = add_layers(self, dimensions)

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.relu(layer(x))
        return self.layers[-1](x)
