import torch

import toumei.objectives.objective
import toumei.probe as probe
from toumei.objectives import Pipeline
import toumei.objectives.atoms as obj
import toumei.objectives.operations as ops
import toumei.parameterization as param
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
alexNet = models.alexnet()
probe.print_modules(alexNet)

n = obj.Neuron("features.0:0:0")
toumei.objectives.objective.freezeModel(alexNet)
n.attach(alexNet)
x = torch.rand((1, 3, 512, 512), requires_grad=True)
print(n, n.hook, n())
alexNet(x)
print(n, n.hook, n())
