import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(0)

class Net_54(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 5)
        self.fc2 = nn.Linear(5, 4)
        self.fc3 = nn.Linear(4, 1)

    def forward(self, x):
        # F.relu doesn't work here (best loss is 0.17).
        # F.sigmoid does, but takes ~20x longer.
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))
        x = self.fc3(x)
        return x

class Net_9(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 9)
        self.fc2 = nn.Linear(9, 1)

    def forward(self, x):
        # F.relu doesn't work here (best loss is 0.17).
        # F.sigmoid does, but takes ~4x longer.
        x = F.silu(self.fc1(x))
        x = self.fc2(x)
        return x

data = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
labels = torch.tensor([0.0, 1.0, 1.0, 0.0]).view(-1, 1)

def train(net, criterion, optimizer):
    nsteps = 0
    last_loss = None
    while True:
        nsteps += 1
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        # Sometimes loss stalls briefly, so don't simply check it's lower than
        # last step. Check it's lower than 1k steps ago.
        val = loss.item()
        if nsteps % 1000 == 0:
            print(val)
            if last_loss is not None and val >= last_loss:
                print(f'{nsteps} steps')
                print(output)
                break
            last_loss = val

def main():
    net = Net_9()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    train(net, criterion, optimizer)


if __name__ == '__main__':
    main()
