import torch
from sacred import Ingredient
from torch.utils.data import DataLoader

from mnist_dataset import MNISTDataset

data_ingredient = Ingredient('dataset')


@data_ingredient.config
def cfg():
    length = 6500
    n_numbers = 2


@data_ingredient.capture
def generate_data(length, n_numbers, device):
    dataset = MNISTDataset(device=device, n_numbers=n_numbers, length=length)
    return dataset