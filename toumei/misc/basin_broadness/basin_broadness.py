import torch
from torch import nn

from toumei.models import SimpleMLP
from base import MLPWrapper

if __name__ == '__main__':
    model = SimpleMLP(2, 8, 4, 2, 1)
    inputs = torch.randn(size=(1024, 2), dtype=torch.float)
    labels = model(inputs)
    w = MLPWrapper(model, inputs, labels)

    print(w[0].orthogonal_parameters)
