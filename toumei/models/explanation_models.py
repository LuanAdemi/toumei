import torch
import torch.nn as nn


class LinearExplanationModel(nn.Module):
    def __init__(self, size=10):
        super(LinearExplanationModel, self).__init__()

        # these are the parameters for optimization
        self.effect_vector = nn.Parameter(torch.randn(size))

    def forward(self, z):
        return self.effect_vector[0] + torch.dot(self.effect_vector[0:], z)

    def __str__(self):
        return f"g(x) = {self.effect_vector[0]:.2f}" + \
               " ".join([f"{self.effect_vector[i]:.2f} x_{i}" for i in range(1, len(self.effect_vector))])
