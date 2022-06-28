import sys
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
from torchvision.transforms import ToTensor

sys.path.append("../../")
from experiments.research.patch_model import PatchedModel
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
data = tv.datasets.MNIST(root="./data", download=True, train=True, transform=ToTensor())
dataLoader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)
loss_fc = nn.MSELoss()

network = PatchedModel().to(device)
network.load_state_dict(torch.load("models/patched_model.pth", map_location=device))


# a broadness measurer for patched models
class BroadnessMeasurer(object):
    def __init__(self, model):
        self.model = model

    def measure_broadness(self, std_list, num_iters):
        losses_measured = []
        for std in std_list:
            losses_measured.append(self.measure_broadness_for_std(std, num_iters))
        sns.heatmap(losses_measured)
        plt.show()
        return losses_measured

    def measure_broadness_for_std(self, std, num_iters=1):
        loss_measured = []
        for i in range(num_iters):
            print("Standard deviation: ", std, ", num_iter: ", i + 1)
            altered_model = deepcopy(self.model)
            self.alter_params(altered_model, std)
            loss_measured.append(self.get_loss(altered_model))
        return loss_measured

    def get_loss(self, model):
        loss_train = []
        picture_storage = []
        for h, (element, label) in enumerate(dataLoader):
            # if batch is not complete
            if element.shape[0] != 64:
                break

            element = element.to(device)
            label = label.to(device)
            # two images needed for input
            if len(picture_storage) == 0:
                picture_storage.append((element, label))
                continue
            else:
                (prev_element, prev_label) = picture_storage.pop()
                result = prev_label + label
                inp = torch.cat((prev_element.view(-1, 28 * 28), element.view(-1, 28 * 28)), dim=1)
                predicted_result = model(inp)
                loss = loss_fc(predicted_result.view(64), result.float())
                loss.backward()
                loss_train.append(loss.item())
        return np.mean(loss_train)

    # alter every parameter by a std. this should be run multiple times per std
    # the given model should be a patched model
    def alter_params(self, model, standard_deviation=0.01):
        for named_params in zip(model.patched_m.named_parameters()):
            (key, value) = named_params[0]

            # it might be interesting to differ weight std and bias std
            if 'weight' in key:
                for neuron_weights in value:
                    for i in range(len(neuron_weights)):
                        current_weight = neuron_weights[i]
                        neuron_weights[i] = np.random.randn() * standard_deviation + neuron_weights[i]
            elif 'bias' in key:
                pass
        return model


measurer = BroadnessMeasurer(network)
std_list = [x * 0.01 for x in range(10)]
print(measurer.measure_broadness(std_list, 10))
