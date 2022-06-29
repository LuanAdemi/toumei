import random

import torch
from torch.utils.data import Dataset


def _binary(x, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()


class OneHotEncodingDataset(Dataset):
    def __init__(self, length):
        super(OneHotEncodingDataset, self).__init__()
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        a = random.randint(0, 7)
        b = random.randint(0, 7)
        one_hot_a = torch.zeros((1, 8))
        one_hot_a[0, a] = 1
        one_hot_b = torch.zeros((1, 8))
        one_hot_b[0, b] = 1

        data = torch.cat((one_hot_a, one_hot_b), dim=1)

        return data, (a, b)
