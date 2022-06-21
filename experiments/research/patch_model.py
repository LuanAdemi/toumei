import torch
import torch.nn as nn
from toumei.models import SimpleMLP
from torchvision.transforms import ToTensor

import torchvision as tv

device = torch.device("cuda")
data = tv.datasets.MNIST(root="./data", download=True, train=True, transform=ToTensor())
dataLoader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)


class PatchedModel(nn.Module):
    def __init__(self):
        super(PatchedModel, self).__init__()

        self.m_1 = SimpleMLP(28*28, (28*28)//2, 10)
        self.m_2 = SimpleMLP(28*28, (28*28)//2, 10)

        self.mlp = SimpleMLP(20, 4, 1)

        self.m_1.load_state_dict(torch.load("mnist_model.pth"))
        self.m_2.load_state_dict(torch.load("mnist_model.pth"))

        self.mlp.load_state_dict(torch.load("addition_model.pth"))

    def forward(self, x_1, x_2):
        x_1 = torch.softmax(self.m_1(x_1), dim=1)
        x_2 = torch.softmax(self.m_2(x_2), dim=1)

        x = torch.cat((x_1, x_2), dim=-1)

        x = self.mlp(x)
        return x


model = PatchedModel().to(device)

i_1, i_2 = 42, 69

print(data[i_1][1], data[i_2][1])

x_1 = data[i_1][0].view(-1, 28*28).to(device)
x_2 = data[i_2][0].view(-1, 28*28).to(device)

print(model(x_1, x_2))
