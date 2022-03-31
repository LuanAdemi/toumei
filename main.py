import torch

import toumei.probe as p
import toumei.objectives as objective
import torch.nn as nn
import torchvision.models as models

model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid()
)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.f1 = nn.Linear(2, 8)
        self.relu = nn.ReLU()
        self.f2 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.f1(x)
        x = self.relu(x)
        x = self.f2(x)
        return self.sigmoid(x)


model2 = Net()
inception = models.alexnet()
p.print_modules(inception)
objective = objective.Sequential(
    objective.Layer("features.0"),
    objective.Neuron("features.0:0:0")
)

objective.attach(inception)
inception(torch.rand((1, 3, 128, 128)))
print(objective.activation)
