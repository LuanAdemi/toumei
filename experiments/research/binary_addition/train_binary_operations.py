import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.research.binary_addition.binary_task import AddOperator
from experiments.research.binary_addition.n_numbers_datasets import BinaryDataset, OneHotEncodingDataset, MNISTDataset
from toumei.models import SimpleMLP


class BinaryOperationsHandler(object):
    def __init__(self, data="onehot", task=AddOperator(), numbers_to_process=2, ep=10, len_dataset=2 ** 15,
                 input_size=3,
                 batch_size=1, lr=1e-5, hid_dimensions=(16, 4)):
        self.data = data
        self.task = task
        self.numbers_to_process = numbers_to_process
        self.ep = ep
        self.len_dataset = len_dataset

        self.input_size = 0

        if data == "binary":
            self.input_size = input_size
        elif data == "onehot":
            self.input_size = input_size + 1
        elif data == "mnist":
            self.input_size = 784

        self.batch_size = batch_size
        self.lr = lr
        self.hid_dimensions = hid_dimensions

        if data == "binary":
            self.data = BinaryDataset(self.len_dataset, bits=input_size, numbers_to_process=numbers_to_process)
        elif data == "onehot":
            self.data = OneHotEncodingDataset(self.len_dataset, max_number=input_size,
                                              numbers_to_process=numbers_to_process)
        elif data == "mnist":
            self.data = MNISTDataset(self.len_dataset, numbers_to_process)

        self.dataLoader = torch.utils.data.DataLoader(self.data, batch_size=batch_size)
        self.device = torch.device("cuda")
        self.loss_fc = torch.nn.MSELoss()

    def train(self, save_model=False, save_plot=False, iteration=1):
        network = SimpleMLP(self.numbers_to_process * self.input_size, *self.hid_dimensions, 1).to(self.device)
        opt = torch.optim.Adam(lr=self.lr, params=network.parameters())
        global_losses = []
        for i in range(self.ep):
            loss_train = []

            if i % self.task.mvg_change_epochs == 0 and i > self.task.no_mvg_epochs:
                self.task.change_params(random_goals=False)

            for h, (element, label) in enumerate(self.dataLoader):
                element = element.to(self.device)
                result = self.task.operate(label)
                predicted_result = network(element)
                opt.zero_grad()
                loss = self.loss_fc(predicted_result.view(-1), result.float().to(self.device))
                loss.backward()
                opt.step()

                loss_train.append(loss.item())
                global_losses.append(loss.item())
            print('TRAIN: EPOCH %d: BATCH %d: LOSS: %.4f PARAMS: %s' %
                  (i, h, np.mean(loss_train), str(self.task.mvg_parameters)))

        len_losses = (self.len_dataset // self.batch_size) * self.ep
        x1 = np.linspace(1, len_losses, num=len_losses)
        plt.plot(x1, global_losses)
        plt.show()

        if save_plot:
            plt.savefig("plots/" + self.task.get_name())

        if save_model:
            torch.save(network.state_dict(),
                       "models/mnist_add_2_with_inc_model_size/" + "mnist_add_2_with_inc_model_size" + str(
                           iteration) + ".pth")

    def props_to_string(self):
        info = "----------------------------------------\n"
        info += "Training infos: \n"
        info += "Input Encoding: " + str(self.data) + "\n"
        info += "Task: " + self.task.get_name() + "\n"
        info += "Numbers given to the model: " + str(self.numbers_to_process) + "\n"
        info += "Episodes: " + str(self.ep) + "\n"
        info += "Size of the dataset: " + str(self.len_dataset) + "\n"
        info += "The size of the vector representing an input number: " + str(self.input_size) + "\n"
        info += "Batch size: " + str(self.batch_size) + "\n"
        info += "Learning rate: " + str(self.lr) + "\n"
        info += "Hidden sizes of the model: " + str(self.hid_dimensions)
        return info


task = AddOperator(mvg=True, mvg_size=2, mvg_change_epochs=2, no_mvg_epochs=0)
for i in range(50):
    handler = BinaryOperationsHandler(data="mnist", task=task, numbers_to_process=2, ep=250, len_dataset=60000,
                                      input_size=4, batch_size=32,
                                      hid_dimensions=(384 + (256 // 25) * i, 384 + (256 // 25) * i, 128, 4))
    print(handler.props_to_string())
    handler.train(save_plot=True, save_model=True, iteration=i)
