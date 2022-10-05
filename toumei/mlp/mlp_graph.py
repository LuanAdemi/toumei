import itertools
import math

from toumei.parents import ModelGraph


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

    def _get_weights(self):
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

    def _build_graph(self):
        """
        Iteratively build the graph from the MLP using the weight matrices

        This uses the absolute values of the weight between two connected neurons as seen in
        https://arxiv.org/pdf/2110.08058.pdf
        """
        # get the named parameter weights
        weights = self._get_weights()

        current_layer = 0

        for key in weights.keys():
            # get the current named parameter weight
            value = weights[key]

            current_layer += 1
            for current_neuron in range(value.shape[1]):
                current_node = f"layer{current_layer}:{current_neuron}"
                # iterate over every sub node
                for next_neuron in range(value.shape[0]):
                    next_node = f"layer{current_layer + 1}:{next_neuron}"

                    # add an edge between the two nodes using the absolute value of the parameter weight as the
                    # edge weight
                    if not self.has_node(current_node):
                        self.add_node(current_node, layer=current_layer)
                    if not self.has_node(next_node):
                        self.add_node(next_node, layer=current_layer + 1)
                    w = value[next_neuron, current_neuron].detach().abs().item()
                    v = value[next_neuron, current_neuron].detach().item()
                    num_edges_prev = len(self.edges.items())
                    self.add_edge(current_node, next_node, weight=w, title=v, value=w)
