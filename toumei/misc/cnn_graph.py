import networkx as nx
import torch.nn as nn


class CNNGraph(nx.Graph):
    """
    Build a graph from an MLP. This enables us to perform graph algorithms on it.

    One important usage is the calculation of the network modularity using the model weights as edge weights.
    """
    def __init__(self, model: nn.Module):
        """
        Initialize a new CNNgraph object for converting a pytorch convolutional model to a graph
        we can perform graph algorithms on

        :param model: the model
        """
        super(CNNGraph, self).__init__()

        self.model = model
        self.build_graph()

    def __str__(self):
        return f"CNNGraph()"

    def build_graph(self):
        return NotImplementedError
