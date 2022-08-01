import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.research.binary_addition.n_numbers_datasets import OneHotEncodingDataset
from toumei.models import SimpleMLP

device = torch.device("cuda")
batch_size = 16
beltalowda = OneHotEncodingDataset(45000, elements=8)
dataLoader = torch.utils.data.DataLoader(beltalowda, batch_size=batch_size)

ep = 100
network = SimpleMLP(16, 8, 4, 2, 1).to(device)
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
        elif current_task == 1:
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

#torch.save(network.state_dict(), "models/binary_addition_model_961.pth")
x1 = np.linspace(1, len(global_losses), num=len(global_losses))
plt.plot(x1, global_losses)
plt.show()
