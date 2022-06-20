import networkx as nx
import torch.nn as nn

from sklearn.cluster import SpectralClustering


class ModelGraph(nx.Graph):
    """
    Build a graph from an MLP. This enables us to perform graph algorithms on it.

    One important usage is the calculation of the network modularity using the model weights as edge weights.
    """
    def __init__(self, model: nn.Module):
        """
        Initialize a new ModelGraph object for converting a pytorch model to a graph we can perform graph algorithms on

        :param model: the model
        """
        super(ModelGraph, self).__init__()

        self.model = model
        self._build_graph()

    def __str__(self):
        return f"ModelGraph()"

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
        Build the graph for the model
        """
        return NotImplementedError

    def _spectral_clustering(self, n_clusters=8):
        """
        Performs network-wide spectral clustering.

        TODO: Convert the simple cluster labels to a networkx node list

        :return: the cluster labels
        """

        adj_matrix = nx.to_numpy_matrix(self)
        node_list = list(self.nodes())

        '''Spectral Clustering'''
        clusters = SpectralClustering(affinity='precomputed', assign_labels="discretize", random_state=0,
                                      n_clusters=n_clusters).fit_predict(adj_matrix)

        return clusters

    def get_model_modularity(self):
        """
        Calculate the best-case modularity of the model by calculating the graph modularity

        The values range from [-1, 2.1].

        TODO: Use spectral clustering here, as seen in the original paper

        :return: the model modularity
        """

        # perform spectral clustering for partitioning the graph
        self._spectral_clustering()

        # OBSOLETE: greedily find the best graph partition (community) by maximizing modularity
        best_partition = nx.algorithms.community.greedy_modularity_communities(self)

        # calculate the modularity for the given partition
        return nx.algorithms.community.modularity(self, best_partition)
