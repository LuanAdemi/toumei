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
        if model is not None:
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

    def _spectral_clustering(self, n_clusters=32, gamma=1.):
        """
        Performs network-wide spectral clustering.

        :return: the cluster labels
        """

        adj_matrix = nx.to_numpy_matrix(self)
        node_list = list(self.nodes())

        clusters = SpectralClustering(eigen_solver='arpack', n_init=100, affinity='precomputed', assign_labels="kmeans",
                                      n_clusters=n_clusters, gamma=gamma).fit_predict(adj_matrix)

        communities = [set() for _ in range(n_clusters)]

        for i, node in enumerate(node_list):
            label = clusters[i]
            communities[label].add(node)

        communities = [frozenset(i) for i in communities]

        return communities, clusters

    def get_model_modularity(self, n_clusters=8, resolution=1, method="louvain", communities=None):
        """
        Calculate the best-case modularity of the model by calculating the graph modularity

        The values range from [-1, 2.1].

        :return: the model modularity
        """
        G = nx.Graph(nx.adjacency_matrix(self))
        if communities is None:
            if method == "spectral":
                communities, clusters = self._spectral_clustering(n_clusters, gamma=resolution)
                return nx.algorithms.community.modularity(self, communities=communities, resolution=resolution)
            elif method == "greedy":
                communities = nx.community.greedy_modularity_communities(G, resolution=resolution)
                return nx.algorithms.community.modularity(G, communities=communities, resolution=resolution)
            elif method == "louvain":
                communities = nx.community.louvain_communities(G, resolution=resolution)
                return nx.algorithms.community.modularity(G, communities=communities, resolution=resolution)
            else:
                raise Exception("Not a valid clustering method")
