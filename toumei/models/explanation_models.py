import torch
import torch.nn as nn


class MappingFunction(object):
    """
    A function that maps 'interpretable inputs' to the original input vectors
    """
    def __init__(self):
        super(MappingFunction, self).__init__()

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x: torch.Tensor):
        return NotImplementedError


class PatchingMapper(MappingFunction):
    """
    A simple patching mapper, which maps a binary vector z to the original output vector,
    where 1 means the feature is present and 0 means it's not
    """
    def __init__(self, original_input):
        super(PatchingMapper, self).__init__()
        self.original_input = original_input

    def forward(self, x: torch.Tensor):
        return self.original_input[x]


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
