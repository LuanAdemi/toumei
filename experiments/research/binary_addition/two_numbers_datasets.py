import random

import torch
from torch.utils.data import Dataset

device = torch.device("cuda")


def _binary(x, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()


class OneHotEncodingDataset(Dataset):
    """
    This dataset provides two random one-hot encoded numbers
    between 0 and 7 as one tensor per item.
    """

    def __init__(self, elements, length=-1):
        super(OneHotEncodingDataset, self).__init__()
        self.elements = elements


        self.data = []

        for i in range(length):
            a = random.randint(0, elements-1)
            b = random.randint(0, elements - 1)
            one_hot_a = torch.zeros((1, self.elements))
            one_hot_a[0, a] = 1
            one_hot_b = torch.zeros((1, self.elements))
            one_hot_b[0, b] = 1

            d = torch.cat((one_hot_a, one_hot_b), dim=1)

            self.data.append(((d + torch.normal(mean=0, std=.05, size=d.shape)).to(device),
                            (torch.FloatTensor([a]).to(device), torch.FloatTensor([b]).to(device))))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):

        return self.data[item]


class BinaryDataset(Dataset):
    """
    This dataset provides two random bit-numbers as one tensor per item.
    """

    def __init__(self, length, bits=3):
        super(BinaryDataset, self).__init__()
        self.length = length
        self.bits = bits

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        num_a = random.randint(0, 2 ** self.bits - 1)
        num_b = random.randint(0, 2 ** self.bits - 1)
        a = torch.IntTensor([num_a])
        b = torch.IntTensor([num_b])
        bin_a = _binary(a, self.bits)
        bin_b = _binary(b, self.bits)

        data = torch.cat((bin_a, bin_b), dim=1)

        return data, (a, b)
