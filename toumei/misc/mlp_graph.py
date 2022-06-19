import networkx as nx
import torch.nn as nn
import itertools


def peek(it):
    first = next(it)
    return first, itertools.chain([first], it)


class MLPGraph(nx.Graph, nn.Module):
    """
    Build a graph from an MLP. This enables us to perform graph algorithms on it.

    One important usage is the calculation of the network modularity using the model weights as edge weights.
    """
    def __init__(self, model: nn.Module):
        super(MLPGraph, self).__init__()

        self.model = model
        self.build_graph()

    def __str__(self):
        return f"MLPGraph()"

    def get_weights(self):
        """
        Retrieve the weights from the model and pack them into a dictionary

        :return: a dictionary containing the weights
        """
        named_params = self.model.named_parameters()

        weights = {}

        for (key, value) in named_params:
            if 'weight' in key:
                weights[key] = value

        return weights

    def build_graph(self):
        """
        Iteratively build the graph from the model using the weight matrices
        """
        weights = self.get_weights()

        iterator = iter(weights.items())

        while True:
            key, value = next(iterator)
            try:
                (next_key, next_value), iterator = peek(iterator)
                for current_neuron in range(value.shape[1]):
                    current_node_name = key.split(".")[0] + ":" + str(current_neuron)

                    for next_neuron in range(value.shape[0]):
                        next_node_name = next_key.split(".")[0] + ":" + str(next_neuron)
                        self.add_edge(current_node_name, next_node_name,
                                       weight=value[next_neuron, current_neuron].detach().item())
            except StopIteration:
                break

    def get_model_modularity(self):
        """
        Calculate the best-case modularity of the model by calculating the graph modularity

        :return: the model modularity
        """
        # greedily find the best graph partition (community) by maximizing modularity
        best_partition = nx.algorithms.community.greedy_modularity_communities(self)

        # calculate the modularity for the given partition
        return nx.algorithms.community.modularity(self, best_partition)

    def forward(self, x):
        return self.model(x)

