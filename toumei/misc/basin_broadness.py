import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam

import seaborn as sns

import tqdm


class BasinVolumeMeasurer(object):
    def __init__(self, model, data):
        self.model = model
        self.parameters = nn.utils.parameters_to_vector(model.parameters())
        self.data = data

    def forward_pass(self):
        return self.model(self.data)

    def calculate_inner_products(self):
        """
        Calculates the L2 inner products of the features over the training set.

        This is done by computing the right term of the hessian decomposition.
        We first build the computation graph by performing a full batch forward pass,
        then accumulating the gradients df(x,θ)/dθ for every x.

        The final matrix is build by calculating the corresponding dot products.

        :return: the l2 inner product matrix
        """
        out = self.forward_pass()
        n, outdim = out.shape
        p_size = len(self.parameters)

        grads = []

        for i in range(n):
            self.model.zero_grad()
            out[i].backward(retain_graph=True)
            p_grad = torch.tensor([], requires_grad=False)
            for p in self.model.parameters():
                p_grad = torch.cat((p_grad, p.grad.reshape(-1)))  # df(x,θ)/dθ
            grads.append(p_grad)

        grads = torch.stack(grads)
        inner_products = torch.zeros(size=(p_size, p_size))

        for j in range(len(self.parameters)):
            for k in range(len(self.parameters)):
                """
                Computes the dot product for 1D tensors. 
                For higher dimensions, sums the product of elements from input 
                and other along their last dimension.
                """
                inner_products[j, k] = torch.inner(grads[:, j], grads[:, k])

        return inner_products


class DummyModel(nn.Module):
    """
    The little example presented in the document
    """
    def __init__(self):
        super(DummyModel, self).__init__()

        # these are the parameters for optimization
        self.param = nn.Parameter(torch.tensor([1, 1, 1], dtype=torch.float))

    def forward(self, z):
        return self.param[0] + self.param[1] * z + self.param[2] * torch.cos(z)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.fc1 = nn.Linear(2, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.fc1(x))


class OrDataset(Dataset):
    def __init__(self):
        self.data = []
        self.labels = []

        self.data = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
        self.labels = [[0.], [1.], [1.], [1.]]

        self.data = torch.tensor(self.data)
        self.labels = torch.tensor(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.labels[item]


def trainOrModel():
    model = Model()
    dataset = OrDataset()
    dataloader = DataLoader(dataset, batch_size=1)

    optimizer = Adam(params=model.parameters(), lr=3e-3)
    loss_func = nn.MSELoss()

    for i in tqdm.trange(5000):
        for (inp, labels) in dataloader:
            out = model(inp)
            loss = loss_func(out, labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    print(loss.item())

    return model, dataset.data


if __name__ == '__main__':
    model, inputs = trainOrModel()
    measurer = BasinVolumeMeasurer(model, inputs)
    inner_products = measurer.calculate_inner_products()
    sns.heatmap(data=inner_products, square=True)
    plt.show()

