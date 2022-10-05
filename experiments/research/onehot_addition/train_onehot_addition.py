import numpy as np
import torch
from torch import nn
from toumei.models import SimpleMLP

device = torch.device("cpu")
batch_size = 8

Xs = torch.Tensor([[0., 0.],
               [0., 1.],
               [1., 0.],
               [1., 1.]])

y = torch.Tensor([0., 1., 1., 0.]).reshape(Xs.shape[0], 1)

ep = 40000
network = SimpleMLP(2, 16, 1, activation=nn.Sigmoid()).to(device)
loss_fc = torch.nn.MSELoss()
opt = torch.optim.Adam(lr=1e-3, params=network.parameters())

current_task = 0

a = 1
b = 1

all_losses = []
current_loss = 0
plot_every = 50

for i in range(ep):
    loss_train = []

    predicted_result = network(Xs)

    loss = loss_fc(predicted_result, y)
    loss.backward()
    opt.step()
    opt.zero_grad()

    # append to loss
    current_loss += loss

    if i % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        print(current_loss / plot_every)
        current_loss = 0


        # print progress
    if i % 500 == 0:
        print(f'Epoch: {i} completed')

torch.save(network.state_dict(), "../../../toumei/misc/basin_broadness/xor16.pth")
