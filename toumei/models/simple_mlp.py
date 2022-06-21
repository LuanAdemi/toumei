import torch.nn as nn


class SimpleMLP(nn.Module):
    layers = []

    def __init__(self, *dimensions):
        super(SimpleMLP, self).__init__()

        for i, (dimension) in enumerate(dimensions):
            fc = nn.Linear(dimension, dimensions[i+1])
            setattr(self, f"fc{i}", fc)
            self.layers.append(fc)
            if len(dimensions) == i+2:
                break

        self.relu = nn.ReLU()

    def forward(self, x):
        for layer in layers:
            x = self.relu(layer(x))
        return x