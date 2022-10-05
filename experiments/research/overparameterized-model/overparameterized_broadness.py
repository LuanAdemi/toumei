import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from toumei.misc.model_broadness import BroadnessMeasurer

class OurBroadnessMeasurer(BroadnessMeasurer):
    # The parent class is basically the same as this, but with
    #
    #     self.dataLoader = DataLoader(dataset, batch_size=len(dataset))
    #
    # instead. Which runs fine but gets different results on the input we use.
    # When calculating loss, it puts every data point in a single batch instead
    # of enumerating over them individually. I don't really know why this is
    # different (MSELoss on a batch should be the same as the mean of MSELosses
    # on single elements?), and I don't know if one is more correct than the
    # other.
    def __init__(self, model, dataset, loss_func, gt_func=None):
        self.model = model
        self.dataLoader = dataset
        self.device = torch.device("cpu")

        self.loss_func = loss_func

        self.gt_func = gt_func

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
    torch.manual_seed(0)

    net = Net_9()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    train(net, criterion, optimizer)

    measurer = OurBroadnessMeasurer(net,
                                    list(zip(data, labels)),
                                    torch.nn.MSELoss())
    _, deltas = measurer.run([x * 0.0001 for x in range(10)], num_itrs=20)
    print(deltas)

if __name__ == '__main__':
    main()
