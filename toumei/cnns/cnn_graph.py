import torch.linalg

from toumei.general.model_graph import ModelGraph
import itertools

def peek(it):
    """
    Peek at the next item of an iterator

    This is done by requesting the next item from the iterator and immediately pushing it back

    :param it: the iterator
    :return: the next element, the reset iterator
    """
    first = next(it)
    return first, itertools.chain([first], it)


class CNNGraph(ModelGraph):
    """
    Build a graph from an CNN. This enables us to perform graph algorithms on it.

    One important usage is the calculation of the network modularity using the model weights as edge weights.
    """

    def __str__(self):
        return f"CNNGraph()"

    def _get_weights(self):
        named_params = self.model.named_parameters()

        weights = {}

        for (key, value) in named_params:
            if 'weight' in key and 'conv' in key:
                weights[key] = value

        return weights

    def _build_graph(self):
        weights = self._get_weights()

        iterator = iter(weights.items())

        current_layer = 0

        while True:
            try:
                # get the current named parameter weight
                key, value = next(iterator)
                current_layer += 1
                for current_channel in range(value.shape[1]):
                    current_node = f"layer{current_layer}:{current_channel}"
                    # iterate over every sub node
                    for next_channel in range(value.shape[0]):
                        next_node = f"layer{current_layer + 1}:{next_channel}"

                        self.add_node(current_node, layer=current_layer)
                        self.add_node(next_node, layer=current_layer + 1)

                        self.add_edge(current_node, next_node,
                                         weight=torch.linalg.matrix_norm(
                                             value[next_channel, current_channel], ord=1).detach().item())

            except StopIteration:
                break
