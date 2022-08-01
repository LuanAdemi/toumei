import sys
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sacred import Experiment
from sacred.observers import MongoObserver

from experiments.research.binary_addition.binary_task import AddOperator
from experiments.research.binary_addition.n_numbers_datasets import MNISTDataset
from toumei.models import SimpleMLP, SimpleCNN

sys.path.append("../../")
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda")
loss_fc = nn.MSELoss()


# a broadness measurer for patched models
class BroadnessMeasurer(object):
    def __init__(self, model, dataLoader, task):
        self.model = model.to(device)
        self.dataLoader = dataLoader
        self.task = task

    def measure_broadness(self, _run, std_list, num_iters=5, save_plot=False, plot_name=''):
        losses_measured = []
        loss_deltas = []
        for std in std_list:
            if std == 0:
                # for std 0 always measure one time since there is only one possible outcome
                outcome = self.measure_broadness_for_std(std)
            else:
                outcome = self.measure_broadness_for_std(std, num_iters)

            if len(losses_measured) != 0:
                delta = np.mean(outcome) - np.mean(losses_measured[-1])
                loss_deltas.append(delta)
            losses_measured.append(outcome)
            _run.log_scalar("loss", np.mean(outcome), std)
            print("Losses for std = ", str(std), ": ", outcome)

        losses_measured = np.array(losses_measured)
        loss_deltas = np.array(loss_deltas)

        if save_plot:
            # save delta of losses
            x1 = np.linspace(1, len(loss_deltas), num=len(loss_deltas))
            plt.ylim(0, 0.05)
            plt.plot(x1, loss_deltas)
            if plot_name == '':
                plt.savefig('plots/550_1_MVG_deltas.png')
            else:
                plt.savefig(plot_name + "_deltas" + ".png")

            # save losses
            plot_losses = []
            for i, losses in enumerate(losses_measured):
                std = std_list[i]
                for loss in losses:
                    plot_losses.append([std, loss])
            plt.figure(figsize=(15, 8))
            df = pd.DataFrame(columns=["standard_deviation", "loss"], data=plot_losses)
            sns.lineplot(data=df, x='standard_deviation', y='loss')
            if plot_name == '':
                plt.savefig('plots/550_1_MVG_losses.png')
            else:
                plt.savefig(plot_name + ".png")
        return losses_measured, loss_deltas

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

        for h, (element, label) in enumerate(self.dataLoader):
            element = element.to(device)
            label = label.to(device)
            predicted_result = model(element)
            result = self.task.operate(label).to(device)
            loss = loss_fc(predicted_result.view(-1), result.float())
            loss_train.append(loss.item())
        return np.mean(loss_train)

    # alter every parameter by a std. this should be run multiple times per std
    # the given model should be a patched model
    def alter_params(self, model, standard_deviation=0.01):
        if standard_deviation == 0:
            return model
        for named_params in zip(model.named_parameters()):
            (key, value) = named_params[0]

            # it might be interesting to differ weight std and bias std
            if 'weight' in key:
                with torch.no_grad():
                    noise = torch.randn_like(value) * (standard_deviation ** 0.5)
                    value += noise
            elif 'bias' in key:
                pass
        return model

    def get_input_size(self):
        first_layer = next(self.model.named_parameters())
        (layer_name, parameters) = first_layer
        return parameters.size(dim=1)

    def get_influencial_params(self, model, std_list):
        for named_params in zip(model.named_parameters()):
            (key, value) = named_params[0]

            if 'weight' in key:
                for j in range(len(value)):
                    neuron_weights = value[j]
                    for i in range(len(neuron_weights)):
                        # this code runs for every weight
                        current_weight = neuron_weights[i]
                        weight_losses = []
                        for std in std_list:
                            with torch.no_grad():
                                neuron_weights[i] = current_weight + std
                                loss = self.get_loss(model)
                                weight_losses.append(loss)

                        # reset this weight
                        neuron_weights[i] = current_weight
            elif 'bias' in key:
                pass


ex = Experiment("broadness_cnn_test")
ex.observers.append(MongoObserver(url="192.168.188.91:27017"))


@ex.config
def cfg():
    device = torch.device("cuda")
    numbers_to_process = 2
    mvg = True
    no_mvg_epochs = 2
    mvg_change_epochs = 5
    # if you want one specific parameter-group, enter here, else empty list
    mvg_parameters = [1, 1]
    batch_size = 64
    shuffle_dataset = False
    architecture = "CNN"
    net_dimensions = (2, 1)
    model_path = "models/cnn_mnist_add_2/cnn_fixed_mnist_add_2_nomvg.pth"
    standard_deviations = [x * 0.00001 for x in range(20)]
    num_iterations_per_std = 50
    save_plot = True
    plot_path = "plots/cnn_mnist_add_2/cnn_mnist_add_2_nomvg_test"


@ex.automain
def run(_run, device, numbers_to_process, mvg, no_mvg_epochs, mvg_change_epochs, mvg_parameters, batch_size,
        shuffle_dataset,
        architecture, net_dimensions, model_path, standard_deviations, num_iterations_per_std, save_plot, plot_path):
    task = AddOperator(mvg=mvg, mvg_size=numbers_to_process, no_mvg_epochs=no_mvg_epochs,
                       mvg_change_epochs=mvg_change_epochs, mvg_parameters=mvg_parameters)
    all_losses = []
    data = MNISTDataset()
    beltalowda = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle_dataset)

    if architecture == "CNN":
        network = SimpleCNN(*net_dimensions)
    elif architecture == "MLP":
        network = SimpleMLP(*net_dimensions)

    loaded_network = torch.load(model_path, map_location=device)
    network.load_state_dict(loaded_network)
    measurer = BroadnessMeasurer(network, beltalowda, task)

    standard_deviations = standard_deviations

    losses_measured, loss_deltas = measurer.measure_broadness(_run, standard_deviations,
                                                              num_iters=num_iterations_per_std,
                                                              save_plot=save_plot, plot_name=plot_path)

    print(losses_measured)
    all_losses.append(all_losses)

    ex.add_artifact(plot_path + ".png", 'plot.png')

    return float(np.mean(loss_deltas))

    # TODO: top x parameters in change
