import random as rand
from abc import abstractmethod, ABC

import numpy as np
import torch


class Task(ABC):
    """An abstract class that handles different mvg tasks.
    Its abstract method is 'operate', which all tasks have to implement."""

    def __init__(self, mvg=False, mvg_size=5, no_mvg_epochs=2, mvg_change_epochs=5, lower_bound=-1, upper_bound=2):
        self.mvg = mvg
        self.no_mvg_epochs = no_mvg_epochs
        self.mvg_change_epochs = mvg_change_epochs
        self.mvg_parameters = np.ones(mvg_size).tolist()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    @abstractmethod
    def operate(self, *args):
        pass

    @abstractmethod
    def get_name(self):
        return "task"

    def change_params(self):
        for i in range(len(self.mvg_parameters)):
            self.mvg_parameters[i] = rand.randint(self.lower_bound, self.upper_bound)
            print("changed params")


class XorOperator(Task):
    def operate(self, params_list):
        """Takes a batch of float tensors."""
        results = []
        for batch in params_list:
            result = 0
            for param in batch:
                # incoming tensors are float
                result = result ^ param.int()
            results.append(result)
        return torch.Tensor(results)

    def get_name(self):
        return "xor"


class AddOperator(Task):
    def operate(self, params_list):
        """Takes a batch of float tensors."""
        results = []
        for batch in params_list:
            result = 0
            for param in batch:
                # incoming tensors are float
                result += param.int()
            results.append(result)
        return torch.Tensor(results)

    def get_name(self):
        return "add"


class XorAddOperator(Task):
    def operate(self, params_list):
        """Takes a batch of float tensors."""
        results = []
        for batch in params_list:
            result = 0
            for param in batch:
                # incoming tensors are float
                result = result ^ param.int()

            for param in batch:
                result += param.int()

            results.append(result)
        return torch.Tensor(results)

    def get_name(self):
        return "xoradd"
