import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
from torchvision.transforms import ToTensor

sys.path.append("../../")
from experiments.research.patch_model import PatchedModel

device = torch.device("cuda")
data = tv.datasets.MNIST(root="./data", download=True, train=True, transform=ToTensor())
dataLoader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)
loss_fc = nn.MSELoss

network = PatchedModel().to(device)
network.load_state_dict(torch.load("patched_model.pth"))


def get_loss(model):
    loss_train = []
    picture_storage = []
    opt = torch.optim.Adam(lr=0.01, params=model.parameters())
    for h, (element, label) in enumerate(dataLoader):
        # if batch is not complete
        if element.shape[0] != 64:
            break

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
    return np.mean(loss_train)


# alter every parameter by a std. this should be run multiple times per std
# the given model should be a patched model
def alter_params(model, standard_deviation):
    for named_params in zip(model.patched_m.named_parameters()):
        (key, value) = named_params

        # it might be interesting to differ weight std and bias std
        if 'weight' in key:
            value = np.random.randn() * standard_deviation + value
        elif 'bias' in key:
            value = np.random.randn() * standard_deviation + value


print(get_loss(network))
print(get_loss(alter_params(network, 1)))
print(get_loss(alter_params(network, 10)))
