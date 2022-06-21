import torch.nn as nn


class SimpleMLP(nn.Module):
    def __init__(self, inp, out):
        super(SimpleMLP, self).__init__()

        self.fc1 = nn.Linear(inp, inp)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(inp, inp//2)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(inp//2, inp//4)
        self.relu3 = nn.ReLU()

        self.fc4 = nn.Linear(inp//4, out * 20)
        self.relu4 = nn.ReLU()

        self.fc5 = nn.Linear(out * 20, out)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.relu4(self.fc4(x))
        x = self.relu5(self.fc5(x))
        return x