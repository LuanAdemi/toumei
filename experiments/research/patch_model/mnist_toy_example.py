from torchvision.transforms import ToTensor

import torchvision as tv
import torch
import numpy as np
from toumei.models import SimpleMLP

device = torch.device("cuda")
data = tv.datasets.MNIST(root="./data", download=True, train=True, transform=ToTensor())
dataLoader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)

ep = 100
network = SimpleMLP(28*28, (28*28)//2, 10).to(device)
loss_fc = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(lr=0.001, params=network.parameters())

for i in range(ep):
    network.train()
    loss_train = []

    for h, (element, label) in enumerate(dataLoader):
        element = element.to(device).view(-1, 28*28)
        label = label.to(device)
        opt.zero_grad()

        predicted_result = network(element)
        loss = loss_fc(predicted_result, label)
        loss.backward()
        opt.step()
        loss_train.append(loss.item())
        print('TRAIN: EPOCH %d: BATCH %d: LOSS: %.4f' %
              (i, h, np.mean(loss_train)))

torch.save(network.state_dict(), "../models/mnist_model.pth")
