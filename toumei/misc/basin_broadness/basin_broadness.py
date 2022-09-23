import torch
from torch import nn

from toumei.models import SimpleMLP
from base import MLPWrapper

if __name__ == '__main__':
    Xs = torch.Tensor([[0., 0.],
                       [0., 1.],
                       [1., 0.],
                       [1., 1.]])

    y = torch.Tensor([0., 1., 1., 0.]).reshape(Xs.shape[0], 1)

    model = SimpleMLP(2, 4, 1, activation=nn.Sigmoid())
    model.load_state_dict(torch.load("xor_model_2_4_1.pth"))
    w = MLPWrapper(model, Xs, y)

    print(w[1].activation_matrix)
    print(w[1].params.shape)
    print(w[1].orthogonal_basis[0] @ w[1].orthogonal_basis[0].T)
