import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
from torchvision.transforms import ToTensor

from experiments.research.patch_model import PatchedModel

device = torch.device("cuda")
data = tv.datasets.MNIST(root="./data", download=True, train=True, transform=ToTensor())
dataLoader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)
model = PatchedModel().to(device)

opt = torch.optim.Adam(lr=0.001, params=model.parameters())
loss_fc = nn.MSELoss()
ep = 50
picture_storage = []

for i in range(ep):
    loss_train = []

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
            result = prev_label + label
            inp = torch.cat((prev_element.view(-1, 28 * 28), element.view(-1, 28 * 28)), dim=1)
            predicted_result = model(inp)
            opt.zero_grad()
            loss = loss_fc(predicted_result.view(64), result.float())
            loss.backward()
            opt.step()
            loss_train.append(loss.item())
            print('TRAIN: EPOCH %d: BATCH %d: LOSS: %.4f' %
                  (i, h, np.mean(loss_train)))

torch.save(model.state_dict(), "patched_model.pth")

i_1, i_2 = 42, 69

print(data[i_1][1], data[i_2][1])

x_1 = data[i_1][0].view(-1, 28 * 28).to(device)
x_2 = data[i_2][0].view(-1, 28 * 28).to(device)

print(model(torch.cat((x_1, x_2)).flatten()))
