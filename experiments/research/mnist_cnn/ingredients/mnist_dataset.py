import torch

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


class MNISTDataset(Dataset):
    def __init__(self, device=torch.device('cpu'), n_numbers=2, length=6500):
        super(MNISTDataset, self).__init__()

        self.mnist_data = MNIST(root='../data', train=True, transform=ToTensor(), download=True)
        self.mnist_data_dataloader = DataLoader(dataset=self.mnist_data, shuffle=False,
                                                num_workers=8, batch_size=n_numbers)

        self.inputs = []
        self.labels = []

        for batch in self.mnist_data_dataloader:
            i, l = batch

            self.inputs.append(torch.permute(i, (1, 0, 2, 3))[0].to(device))
            self.labels.append(l.to(device))

        self.inputs = self.inputs[:length]
        self.labels = self.labels[:length]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return self.inputs[item], self.labels[item]
