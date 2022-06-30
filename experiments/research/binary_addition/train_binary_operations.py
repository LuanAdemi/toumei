import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.research.binary_addition.binary_task import AddOperator, XorAddOperator
from experiments.research.binary_addition.two_numbers_datasets import BinaryDataset, OneHotEncodingDataset
from toumei.models import SimpleMLP


class BinaryOperationsHandler(object):
    def __init__(self, binary=False, task=AddOperator(), numbers_to_process=2, ep=10, len_dataset=2 ** 15, input_size=3,
                 batch_size=1, lr=1e-3, hid_dimensions=(16, 4)):
        self.binary = binary
        self.task = task
        self.numbers_to_process = numbers_to_process
        self.ep = ep
        self.len_dataset = len_dataset
        self.input_size = input_size if binary else input_size + 1
        self.batch_size = batch_size
        self.lr = lr
        self.hid_dimensions = hid_dimensions

        self.data = BinaryDataset(self.len_dataset, bits=input_size) if binary \
            else OneHotEncodingDataset(self.len_dataset, max_number=input_size)
        self.dataLoader = torch.utils.data.DataLoader(self.data, batch_size=batch_size)
        self.device = "cuda"
        self.loss_fc = torch.nn.MSELoss()

    def train(self, save_model=False, save_plot=False):
        network = SimpleMLP(self.numbers_to_process, *self.hid_dimensions, 1).to(self.device)
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

    def props_to_string(self):
        info = "----------------------------------------\n"
        info += "Training infos: \n"
        info += "Input Encoding: " + ("Binary" if self.binary else "One Hot") + "\n"
        info += "Task: " + self.task.get_name() + "\n"
        info += "Numbers given to the model: " + str(self.numbers_to_process) + "\n"
        info += "Episodes: " + str(self.ep) + "\n"
        info += "Size of the dataset: " + str(self.len_dataset) + "\n"
        info += "The size of the vector representing an input number: " + str(self.input_size) + "\n"
        info += "Batch size: " + str(self.batch_size) + "\n"
        info += "Learning rate: " + str(self.lr) + "\n"
        info += "Hidden sizes of the model: " + str(self.hid_dimensions)
        return info


task = XorAddOperator(mvg=True, mvg_change_epochs=10)
handler = BinaryOperationsHandler(binary=False, task=task, ep=1000, len_dataset=2 ** 8, input_size=32, batch_size=8,
                                  hid_dimensions=(64, 16, 4))
print(handler.props_to_string())
# handler.train(save_model=True)
