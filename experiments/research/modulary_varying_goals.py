import sys

import numpy as np
import torch
import torchvision as tv
from torchvision.transforms import ToTensor

sys.path.append("../../")
from toumei.models import SimpleMLP

device = torch.device("cuda")
data = tv.datasets.MNIST(root="./data", download=True, train=True, transform=ToTensor())
dataLoader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)

ep = 500
network = SimpleMLP(2 * 28 * 28, 28 * 28, 20, 20, 4, 1).to(device)
loss_fc = torch.nn.MSELoss()
opt = torch.optim.Adam(lr=0.01, params=network.parameters())

picture_storage = []

a = 1
b = 1

for i in range(ep):
    loss_train = []

    if (i > 90) and (i % 10 == 0):
        a = np.random.rand(1)[0] * 10
        b = np.random.rand(1)[0] * 10
        opt = torch.optim.Adam(lr=0.001, params=network.parameters())

    for h, (element, label) in enumerate(dataLoader):
        # if batch is not complete
        if element.shape[0] != 64:
            print("--------------------------------------------------------\n")
            continue

        element = element.to(device)
        label = label.to(device)
        # two images needed for input
        if len(picture_storage) == 0:
            picture_storage.append((element, label))
            continue
        else:
            (prev_element, prev_label) = picture_storage.pop()
            result = prev_label * a + label * b
            inp = torch.cat((prev_element.view(-1, 28 * 28), element.view(-1, 28 * 28)), dim=1)
            predicted_result = network(inp)
            opt.zero_grad()
            loss = loss_fc(predicted_result.view(64), result.float())
            loss.backward()
            opt.step()
            loss_train.append(loss.item())
            print('TRAIN: EPOCH %d: BATCH %d: LOSS: %.4f PARAM_A: %.4f PARAM_B: %.4f' %
                  (i, h, np.mean(loss_train), a, b))

torch.save(network.state_dict(), "modular_varying_goals_model.pth")
