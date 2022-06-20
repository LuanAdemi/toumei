import networkx as nx
import torch.nn as nn
import itertools

from sklearn.cluster import SpectralClustering

from toumei.misc.model_graph import ModelGraph


def peek(it):
    """
    Peek at the next item of an iterator

    This is done by requesting the next item from the iterator and immediately pushing it back

    :param it: the iterator
    :return: the next element, the reset iterator
    """
    first = next(it)
    return first, itertools.chain([first], it)


class MLPGraph(ModelGraph):
    """
    Build a graph from an MLP. This enables us to perform graph algorithms on it.

    One important usage is the calculation of the network modularity using the model weights as edge weights.
    """

    def __str__(self):
        return f"MLPGraph()"

    def _build_graph(self):
        """
        Iteratively build the graph from the MLP using the weight matrices

        This uses the absolute values of the weight between two connected neurons as seen in
        https://arxiv.org/pdf/2110.08058.pdf
        """
        # get the named parameter weights
        weights = self._get_weights()

        # an iterator for iterating over the named parameters
        iterator = iter(weights.items())

        while True:
            # get the current named parameter weight
            key, value = next(iterator)
            try:
                # peek at the next named parameter
                (next_key, next_value), iterator = peek(iterator)
                for current_neuron in range(value.shape[1]):
                    current_node = key.split(".")[0] + ":" + str(current_neuron)
                    # iterate over evey sub node
                    for next_neuron in range(value.shape[0]):
                        next_node = next_key.split(".")[0] + ":" + str(next_neuron)

                        # add an edge between the two nodes using the absolute value of the parameter weight as the
                        # edge weight
                        super().add_edge(current_node, next_node,
                                       weight=value[next_neuron, current_neuron].detach().abs().item())
            except StopIteration:
                break
