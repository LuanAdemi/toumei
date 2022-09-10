from copy import deepcopy

import torch
from torch.utils.data import DataLoader
import numpy as np

import tqdm


class BroadnessMeasurer(object):
    """
    Measure the broadness of a model.
    """
    def __init__(self, model, dataset, loss_func, gt_func=None):
        self.model = model
        self.dataLoader = DataLoader(dataset, batch_size=len(dataset))
        self.device = torch.device("cpu")

        self.loss_func = loss_func

        self.gt_func = gt_func

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
                outcome = self.measure_broadness_for_std(std, num_itrs)

            if len(losses_measured) != 0:
                delta = np.mean(outcome) - np.mean(losses_measured[-1])
                loss_deltas.append(delta)
            losses_measured.append(outcome)

        losses_measured = np.array(losses_measured)
        loss_deltas = np.array(loss_deltas)

        if normalize:
            losses_measured = losses_measured - float(np.mean(losses_measured[0]))

        return losses_measured, loss_deltas

    def measure_broadness_for_std(self, std, num_iters=1):
        loss_measured = []
        with tqdm.trange(num_iters) as t:
            t.set_description(f"Measuring Broadness for std={std:.5f}")
            for i in t:
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
            if self.gt_func:
                result = self.gt_func(label).to(self.device)
            else:
                result = label
            loss = self.loss_func(predicted_result.view(-1), result.float())
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
                    noise = torch.randn_like(value) * (standard_deviation ** 0.5)
                    value += noise
            elif 'bias' in key:
                pass
        return model
