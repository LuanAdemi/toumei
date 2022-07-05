import random

import numpy as np
import torch
from torch.utils.data import Dataset


def _binary(x, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).int()


class OneHotEncodingDataset(Dataset):
    """
    This dataset provides two random one-hot encoded numbers
    as one tensor per item.
    """

    def __init__(self, length, max_number, batch=2):
        super(OneHotEncodingDataset, self).__init__()
        self.length = length
        self.max_number = max_number
        self.batch = batch

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        numbers = []
        for i in range(self.batch):
            # numbers as decimal
            a = random.randint(0, self.max_number)
            numbers.append(a)

        result = []
        for i in range(self.batch):
            # numbers as one hot encoded vectors
            one_hot_a = np.zeros((1, self.max_number + 1))
            one_hot_a[0, numbers[i]] = 1
            result.append(one_hot_a)

        result = np.array(result).flatten()

        return torch.Tensor(result), torch.Tensor(numbers)


class BinaryDataset(Dataset):
    """
    This dataset provides two random bit-numbers as one tensor per item.
    """

    def __init__(self, length, bits=3, batch=2):
        super(BinaryDataset, self).__init__()
        self.length = length
        self.bits = bits
        self.batch = batch

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
