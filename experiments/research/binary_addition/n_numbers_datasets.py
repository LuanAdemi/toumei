import random

import numpy as np
import torch
import torchvision as tv
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


def _binary(x, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).int()


class OneHotEncodingDataset(Dataset):
    """
    This dataset provides n random one-hot encoded numbers
    as one tensor per item.
    """

    def __init__(self, length, max_number, numbers_to_process=2):
        super(OneHotEncodingDataset, self).__init__()
        self.length = length
        self.max_number = max_number
        self.numbers_to_process = numbers_to_process

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        numbers = []
        for i in range(self.numbers_to_process):
            # numbers as decimal
            a = random.randint(0, self.max_number)
            numbers.append(a)

        result = []
        for i in range(self.numbers_to_process):
            # numbers as one hot encoded vectors
            one_hot_a = np.zeros((1, self.max_number + 1))
            one_hot_a[0, numbers[i]] = 1
            result.append(one_hot_a)

        result = np.array(result).flatten()

        return torch.Tensor(result), torch.Tensor(numbers)


class BinaryDataset(Dataset):
    """
    This dataset provides n random bit-numbers as one tensor per item.
    """

    def __init__(self, length, bits=3, numbers_to_process=2):
        super(BinaryDataset, self).__init__()
        self.length = length
        self.bits = bits
        self.batch = numbers_to_process

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        numbers = []
        result = []
        for i in range(self.batch):
            # number in decimal
            num_a = random.randint(0, 2 ** self.bits - 1)
            numbers.append(num_a)
            # number as tensor
            a = torch.IntTensor([num_a])
            # number as binary
            bin_a = _binary(a, self.bits)
            result.append(bin_a.tolist())

        return torch.Tensor(result).flatten(), torch.Tensor(numbers).int()


class MNISTDataset(Dataset):

    def __init__(self, length=1000, numbers_to_process=2):
        super(MNISTDataset, self).__init__()
        self.length = length
        self.numbers_to_process = numbers_to_process
        self.data_index = 0

        # loading the data
        self.dataset = tv.datasets.MNIST(root="./data", download=True, train=True, transform=ToTensor())

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        elements = []
        labels = []

        if self.data_index >= len(self.dataset) - 1:
            self.data_index = 0

        for i in range(self.numbers_to_process):
            # data = [(element0, label0), (element1, label1), ...]
            (element, label) = self.dataset[self.data_index + i]
            elements.append(element)
            labels.append(torch.Tensor([label]))

        self.data_index += self.numbers_to_process

        element_tensor = torch.cat(elements)
        label_tensor = torch.cat(labels)

        return torch.flatten(element_tensor), label_tensor
