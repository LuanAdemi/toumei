import random

import numpy as np
import torch
import torchvision as tv
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor


def _binary(x, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).int()


class FullOneHotEncodingDataset(Dataset):

    def __init__(self, max_number, num_vectors):
        super(FullOneHotEncodingDataset, self).__init__()
        self.length = max_number ** num_vectors
        self.max_number = max_number
        self.num_vectors = num_vectors
        self.dataset = []
        # build the dataset
        self.append_vector([], [])
        self.index = 0

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        item = self.dataset[self.index]
        if self.index < self.length - 1:
            self.index += 1
        else:
            self.index = 0
        return item

    def append_vector(self, input_vector, input_number_list):
        for i in range(self.max_number):
            new_number_dec = i
            new_number_hot = np.zeros((1, self.max_number), dtype=np.float32)
            new_number_hot[0][new_number_dec] = 1
            output_vector = np.append(input_vector, new_number_hot)
            output_number_list = np.append(input_number_list, new_number_dec)

            if len(output_vector) >= self.max_number * self.num_vectors:
                self.dataset.append((torch.Tensor(output_vector), torch.Tensor(output_number_list)))
            else:
                self.append_vector(output_vector, output_number_list)


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
    def __init__(self, device=torch.device('cpu'), n_numbers=2):
        super(MNISTDataset, self).__init__()

        self.mnist_data = tv.datasets.MNIST(root='./data', train=True, transform=ToTensor(), download=True)
        self.mnist_data_dataloader = DataLoader(dataset=self.mnist_data, shuffle=False,
                                                num_workers=8, batch_size=n_numbers)

        self.inputs = []
        self.labels = []

        for batch in self.mnist_data_dataloader:
            i, l = batch

            self.inputs.append(torch.permute(i, (1, 0, 2, 3))[0].to(device))
            self.labels.append(l.to(device))

        self.inputs = self.inputs[:6500]
        self.labels = self.labels[:6500]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return self.inputs[item], self.labels[item]


"""
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
"""
