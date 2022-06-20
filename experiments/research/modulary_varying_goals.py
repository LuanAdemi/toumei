from torchvision.transforms import ToTensor

from toumei.models import SimpleMLP
import torchvision as tv
import torch
import numpy as np

device = torch.device("cuda")
data = tv.datasets.MNIST(root="./data", download=True, train=True, transform=ToTensor())
dataLoader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)

ep = 100
network = SimpleMLP(1568, 1).to(device)
loss_fc = torch.nn.MSELoss()
opt = torch.optim.Adam(lr=0.001, params=network.parameters())

picture_storage = []

for i in range(ep):
    loss_train = []
    a = 5.
    b = 6.
    for h, (element, label) in enumerate(dataLoader):
        #if batch is not complete
        if element.shape[0] != 64:
            continue

        element = element.to(device)
        label = label.to(device)
        #two images needed for input
        if len(picture_storage) == 0:
            picture_storage.append((element, label))
            continue
        else:
            (element1, label1) = picture_storage.pop()
            result = label1*a + label*b
            inp = torch.cat((element1, element), dim=-1)
            inp = inp.view(64, 28*28*2)
            predicted_result = network(inp)
            opt.zero_grad()
            loss = loss_fc(predicted_result.view(64), result)
            loss.backward()
            opt.step()
            loss_train.append(loss.item())
            print('TRAIN: EPOCH %d: BATCH %d: LOSS: %.4f' %
                    (i, h, np.mean(loss_train)))
