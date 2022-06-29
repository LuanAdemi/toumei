import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.research.binary_addition.OneHotEncodingDataset import OneHotEncodingDataset
from toumei.models import SimpleMLP

device = torch.device("cuda")
batch_size = 32
beltalowda = OneHotEncodingDataset(69420)
dataLoader = torch.utils.data.DataLoader(beltalowda, batch_size=batch_size)

ep = 10
network = SimpleMLP(16, 8, 4, 1).to(device)
loss_fc = torch.nn.MSELoss()
opt = torch.optim.Adam(lr=1e-3, params=network.parameters())

current_task = 0

a = 1
b = 1

global_losses = []

for i in range(ep):
    loss_train = []

    if i % 2 == 0:
        if current_task == 0:
            a = 1
            b = 1
            current_task = 1
        else:
            a = 1
            b = -1
            current_task = 0

    for h, (element, label) in enumerate(dataLoader):
        element = element.to(device)
        (x1, x2) = label
        result = x1 * a + x2 * b
        predicted_result = network(element)
        opt.zero_grad()
        loss = loss_fc(predicted_result.view(-1), result.float().to(device))
        loss.backward()
        opt.step()

        loss_train.append(loss.item())
        global_losses.append(loss.item())
        print('TRAIN: EPOCH %d: BATCH %d: LOSS: %.4f PARAM_A: %.4f PARAM_B: %.4f' %
              (i, h, np.mean(loss_train), a, b))

x1 = np.linspace(1, 21700, num=21700)
plt.plot(x1, global_losses)
plt.show()
# torch.save(network.state_dict(), "binary_addition_model_great_again_womvg.pth")
