import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib as mpl
import matplotlib.pyplot as plt

from toumei.misc.model_broadness import BroadnessMeasurer

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
                print('Output of fully trained model:')
                print(output)
                break
            last_loss = val

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--graph', action='store_true')

    args = parser.parse_args()

    # Would "generate and print random seed" be useful instead of defaulting to
    # 0?
    torch.manual_seed(args.seed)

    net = Net_9()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    train(net, criterion, optimizer)

    if args.graph:
        steps = [x/100 for x in range(0, 101, 1)]
        points = [[[x, y] for x in steps] for y in steps[::-1]]
        results = net(torch.tensor(points)).detach().numpy()

        fix, ax = plt.subplots()
        im = ax.imshow(results)

        # The range of values will be roughly (epsilon, 1-epsilon), so that's
        # what gets assigned colors by default. It looks fine, but means the
        # colorbar doesn't get values at 0 and 1. set_clim fixes that.
        im.set_clim(0, 1)
        cbar = ax.figure.colorbar(im, ax=ax)

        tick_points = list(range(0, 101, 10))
        tick_vals = [x/100 for x in tick_points]
        ax.set_xticks(tick_points, tick_vals)
        ax.set_yticks(tick_points, tick_vals[::-1])

        plt.show()
    else:
        measurer = BroadnessMeasurer(net,
                                     list(zip(data, labels)),
                                     torch.nn.MSELoss())
        _, deltas = measurer.run([x * 0.0001 for x in range(10)],
                                 num_itrs=10000)
        print(deltas)

if __name__ == '__main__':
    main()
