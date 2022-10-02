import torch
from torch import nn

from toumei.models import SimpleMLP
from base import MLPWrapper

if __name__ == '__main__':
    inputs = torch.Tensor([
        [0., 0.],
        [0., 1.],
        [1., 0.],
        [1., 1.]])

    labels = torch.Tensor([0., 1., 1., 0.]).reshape(inputs.shape[0], 1)

    model = SimpleMLP(2, 4, 1, activation=nn.Sigmoid())
    model.load_state_dict(torch.load("xor_model_2_4_1.pth"))
    w = MLPWrapper(model, inputs, labels)

    orthogonal_layer, eigen_values = w[0].orthogonalise()

    ortho_model = w.orthogonal_model()
    print(ortho_model)
