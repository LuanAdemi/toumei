import networkx as nx
import torch.nn as nn
import itertools

from sklearn.cluster import SpectralClustering


def peek(it):
    """
    Peek at the next item of an iterator

    This is done by requesting the next item from the iterator and immediately pushing it back

    :param it: the iterator
    :return: the next element, the reset iterator
    """
    first = next(it)
    return first, itertools.chain([first], it)


class MLPGraph(nx.Graph):
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

        This uses the absolute values of the weight between two connected neurons as seen in
        https://arxiv.org/pdf/2110.08058.pdf
        """
        # get the named parameter weights
        weights = self.get_weights()

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
                    for next_neuron in range(value.shape[0]):
                        next_node = next_key.split(".")[0] + ":" + str(next_neuron)

                        # add an edge between the two nodes using the absolute value of the parameter weight as the
                        # edge weight
                        self.add_edge(current_node, next_node,
                                       weight=value[next_neuron, current_neuron].detach().abs().item())
            except StopIteration:
                break

    def get_model_modularity(self):
        """
        Calculate the best-case modularity of the model by calculating the graph modularity

        The values range from [-1, 2.1].

        TODO: Use spectral clustering here

        :return: the model modularity
        """

        self.spectral_clustering()

        # greedily find the best graph partition (community) by maximizing modularity
        best_partition = nx.algorithms.community.greedy_modularity_communities(self)

        # calculate the modularity for the given partition
        return nx.algorithms.community.modularity(self, best_partition)

    def spectral_clustering(self, n_clusters=8):
        """
        Performs network-wide spectral clustering.

        :return: the cluster labels
        """

        adj_matrix = nx.to_numpy_matrix(self)
        node_list = list(self.nodes())

        '''Spectral Clustering'''
        clusters = SpectralClustering(affinity='precomputed', assign_labels="discretize", random_state=0,
                                      n_clusters=n_clusters).fit_predict(adj_matrix)

        print(clusters)
        return clusters
