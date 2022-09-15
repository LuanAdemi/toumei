from toumei.models import SimpleCNN
from experiments.research.mnist_cnn.ingredients.mnist_dataset import MNISTDataset

from copy import deepcopy

from torch.utils.data import DataLoader
import numpy as np

import tqdm

import torch

# load the example model
model = SimpleCNN(2, 1)
model.load_state_dict(torch.load("models/model_weights.pth", map_location=torch.device('cpu')))

data = MNISTDataset()


class BroadnessMeasurer(object):
    """
    Measure the broadness of a model.
    """
    def __init__(self, model, dataset):
        self.model = model
        self.dataLoader = DataLoader(dataset, batch_size=len(dataset))
        self.device = torch.device("cpu")

        self.loss_func = torch.nn.MSELoss()

    def run(self, std_list, num_itrs=5, normalize=True):
        """
        Start measuring
        :param std_list: the list of standard deviations used for noise sampling
        :param num_itrs: the number of alterations per standard deviations
        :return: the measured losses and loss deltas
        """
        losses_measured = []
        loss_deltas = []
        for std in std_list:
            if std == 0:
                # for std 0 just measure one time since there is only one possible outcome
                outcome = self.measure_broadness_for_std(std)
            else:
                # measure num_iters times
                outcome = self.measure_broadness_for_std(std, num_itrs)

            if len(losses_measured) != 0:
                # calculate an approximate gradient of the resulting loss curve
                delta = np.mean(outcome) - np.mean(losses_measured[-1])
                loss_deltas.append(delta)
            losses_measured.append(outcome)

        losses_measured = np.array(losses_measured, dtype=object)
        loss_deltas = np.array(loss_deltas)

        if normalize:
            losses_measured = losses_measured - float(np.mean(losses_measured[0]))

        return losses_measured, loss_deltas

    def measure_broadness_for_std(self, std, num_iters=1):
        loss_measured = []
        with tqdm.trange(num_iters) as t:
            t.set_description(f"Measuring Broadness for std={std:.5f}")
            for _ in t:
                altered_model = deepcopy(self.model)
                self.alter_params(altered_model, std)
                loss_measured.append(self.get_loss(altered_model))
                t.set_postfix(mean_loss=np.mean(loss_measured))
        return np.array(loss_measured)

    def get_loss(self, model):
        """
        Performs a forward pass through the model to calculate the loss of the model
        :param model: the model
        :return: the mean of the loss
        """
        loss_train = []

        for h, (element, label) in enumerate(self.dataLoader):
            element = element.to(self.device)
            label = label.to(self.device)
            predicted_result = model(element)

            gt = label[:, 0] + label[:, 1]
            loss = self.loss_func(predicted_result.view(-1), gt.float())
            loss_train.append(loss.item())
        return np.mean(loss_train)

    def alter_params(self, model, standard_deviation=0.01):
        """
        This function creates a copy of the model, applies noise sampled from a normal distribution to the params
        and returns the altered model
        :param model: the model to alter
        :param standard_deviation: the standard deviation used for noise sampling
        :return: the altered model
        """
        if standard_deviation == 0:
            return model
        for named_params in zip(model.named_parameters()):
            (key, value) = named_params[0]

            # it might be interesting to differ weight std and bias std
            if 'weight' in key:
                with torch.no_grad():
                    # add noise sampled from a normal distribution to the weights
                    noise = torch.randn_like(value) * (standard_deviation ** 0.5)
                    value += noise
            elif 'bias' in key:
                pass
        return model


measurer = BroadnessMeasurer(model, data)
print(measurer.run([x * 0.0001 for x in range(10)], num_itrs=20))
