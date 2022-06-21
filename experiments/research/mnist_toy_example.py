from torchvision.transforms import ToTensor

import torchvision as tv
import torch
import numpy as np
import sys

sys.path.append("../../")
from toumei.models import SimpleMLP, SimpleCNN

device = torch.device("cuda")
data = tv.datasets.MNIST(root="./data", download=True, train=True, transform=ToTensor())
dataLoader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)

ep = 100
network = SimpleCNN(1, 10).to(device)
loss_fc = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(lr=0.001, params=network.parameters())

for i in range(ep):
    network.train()
    loss_train = []

    for h, (element, label) in enumerate(dataLoader):
        element = element.to(device)
        label = label.to(device)
        predicted_result = network(element)
        opt.zero_grad()
        loss = loss_fc(predicted_result, label)
        loss.backward()
        opt.step()
        loss_train.append(loss.item())
        print('TRAIN: EPOCH %d: BATCH %d: LOSS: %.4f' %
              (i, h, np.mean(loss_train)))

torch.save(network.state_dict(), "mnist_model.pth")