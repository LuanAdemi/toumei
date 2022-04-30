import numpy as np

import toumei.objectives as obj


class ActivationMap(object):
    """
    This class generates an activation map for the specified unit using the given optimized(!) objective.
    An activation map gives an insight on the (spatial) contribution of a feature in other units.
    """
    def __init__(self, objective: obj.Objective, unit: obj.Atom, shape: tuple):
        """
        Initializes a new activation map object
        :param objective: the optimized objective
        :param unit: the unit of the desired activation map
        """
        self.objective = objective
        self.shape = shape

        self.model = self.objective.model
        self.unit = unit

        self.activation_map = np.zeros(self.shape)

    def plot(self):
        """
        Plots the activation map
        """
