import sys
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
from torchvision.transforms import ToTensor
from experiments.research.binary_addition.two_numbers_datasets import OneHotEncodingDataset

from toumei.models import SimpleMLP

sys.path.append("../../")
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
loss_fc = nn.MSELoss()


# a broadness measurer for patched models
class BroadnessMeasurer(object):
    def __init__(self, model, dataLoader):
        self.model = model
        self.dataLoader = dataLoader

    def measure_broadness(self, std_list, num_iters=1):
        losses_measured = []
        for std in std_list:
            if std == 0:
                # for std 0 always measure on time since there is only one possible outcome
                outcome = self.measure_broadness_for_std(std)
                losses_measured.append(outcome)
                print("Losses for std = ", str(std), ": ", outcome)
            else:
                outcome = self.measure_broadness_for_std(std, num_iters)
                losses_measured.append(outcome)
                print("Losses for std = ", str(std), ": ", outcome)
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
        prev_elements = []
        prev_labels = []
        for h, (element, label) in enumerate(self.dataLoader):
            if len(prev_elements) == 0:
                prev_elements = element
                prev_labels = label
                continue
            element = element.to(device)
            label = label.to(device)
            # TODO: wenn die inp dims probleme machen, view benutzen
            #  (vermutlich aus self.model die input size rausholen, hab jz aber keine lust mehr das zu machen sry)
            inp = torch.stack((element, prev_elements), dim=1)
            predicted_result = model(inp)
            result = torch.FloatTensor([label[0] + prev_labels[0]])
            loss = loss_fc(predicted_result, result)
            loss.backward()
            loss_train.append(loss.item())
        return np.mean(loss_train)

    # alter every parameter by a std. this should be run multiple times per std
    # the given model should be a patched model
    def alter_params(self, model, standard_deviation=0.01):
        for named_params in zip(model.named_parameters()):
            (key, value) = named_params[0]

            # it might be interesting to differ weight std and bias std
            if 'weight' in key:
                for j in range(len(value)):
                    neuron_weights = value[j]
                    for i in range(len(neuron_weights)):
                        current_weight = neuron_weights[i]
                        with torch.no_grad():
                            neuron_weights[i] = np.random.randn() * standard_deviation + neuron_weights[i]
            elif 'bias' in key:
                pass
        return model


# TODO: max number (und optional len) des dataset ans model anpassen, der dataloader wird dann der get_loss
#  func übergeben
#  max number is hier die höchste repräsentierbare zahl im one hot vector, der vector wird also max+1 lang sein
data = OneHotEncodingDataset(2000, max_number=10)
beltalowda = torch.utils.data.DataLoader(data, batch_size=16, shuffle=True)
# TODO: lade die models mit richtigen dimensionen
network = SimpleMLP(1568, 784, 20, 20, 4, 1).to(device)
loaded_network = torch.load("models/trained_model.pth", map_location=device)
network.load_state_dict(loaded_network)

# TODO: erzeuge für jedes model einen measurer
measurer = BroadnessMeasurer(network, beltalowda)
# TODO: wenn diese skripte schnell durchlaufen (sollte es bei one hot eig) gerne sample size erhöhen
standard_deviations = [x * 0.01 for x in range(10)]
print(measurer.measure_broadness(standard_deviations, num_iters=20))
