import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.research.binary_addition.binary_task import AddOperator, XorOperator, XorAddOperator
from experiments.research.binary_addition.two_numbers_datasets import BinaryDataset
from toumei.models import SimpleMLP


class BinaryOperationsHandler(object):
    def __init__(self, task="add", ep=10, len_dataset=2 ** 15, bits=3, batch_size=1, lr=1e-3, hid_dimensions=(16)):
        if task == "add":
            self.task = AddOperator()
        elif task == "xor":
            self.task = XorOperator()
        else:
            self.task = XorAddOperator()

        self.ep = ep
        self.len_dataset = len_dataset
        self.bits = bits
        self.batch_size = batch_size
        self.lr = lr
        self.hid_dimensions = hid_dimensions

        self.data = BinaryDataset(self.len_dataset, bits=bits)
        self.dataLoader = torch.utils.data.DataLoader(self.data, batch_size=batch_size)
        self.device = "cuda"
        self.loss_fc = torch.nn.MSELoss()

    def train_on_binary_numbers(self, save_model=False, save_plot=False):
        network = SimpleMLP(self.bits * 2, *self.hid_dimensions, 1).to(self.device)
        opt = torch.optim.Adam(lr=1e-2, params=network.parameters())
        global_losses = []
        for i in range(self.ep):
            loss_train = []

            for h, (element, label) in enumerate(self.dataLoader):
                element = element.to(self.device)
                (x1, x2) = label
                result = self.task.operate(x1, x2)
                predicted_result = network(element)
                opt.zero_grad()
                loss = self.loss_fc(predicted_result.view(-1), result.float().to(self.device))
                loss.backward()
                opt.step()

                loss_train.append(loss.item())
                global_losses.append(loss.item())
                print('TRAIN: EPOCH %d: BATCH %d: LOSS: %.4f' %
                      (i, h, np.mean(loss_train)))

        len_losses = (self.len_dataset // self.batch_size) * self.ep
        x1 = np.linspace(1, len_losses, num=len_losses)
        plt.plot(x1, global_losses)
        plt.show()

        if save_plot:
            plt.savefig("plots/" + self.task.get_name())

        if save_model:
            torch.save(network.state_dict(), "models/" + self.task.get_name() + "_model.pth")


handler = BinaryOperationsHandler(task="xoradd", ep=1100, len_dataset=2 ** 7, bits=3, hid_dimensions=(16, 8, 4))
handler.train_on_binary_numbers(save_model=True)
