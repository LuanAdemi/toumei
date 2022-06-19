import torch.nn as nn


class SimpleMLP(nn.Module):
    def __init__(self, inp, out):
        super(SimpleMLP, self).__init__()

        self.fc1 = nn.Linear(inp, inp*2)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(inp*2, out*2)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(out*2, out)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        return x