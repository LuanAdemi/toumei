import networkx as nx
import torch.nn as nn

from sklearn.cluster import SpectralClustering


class ModelGraph(nx.DiGraph):
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

    def get_model_modularity(self, n_clusters=8, resolution=1, method="louvain"):
        """
        Calculate the modularity of the model by first finding the best partitioning and then calculating the graph
        modularity.

        :param n_clusters The number of clusters used for cluster-based partitioning methods
        :param resolution The resolution for calculating the graph modularity (see the Resolution Limit Problem)
        :param method The partitioning method (spectral, greedy, louvain[preferred])

        :return: the model modularity
        """
        if method != "spectral":
            self.__class__ = nx.Graph

        if method == "spectral":
            """
            Spectral clustering for graph partitioning. This needs a fix number of clusters.
            """
            communities, clusters = self._spectral_clustering(n_clusters, gamma=resolution)

        elif method == "greedy":
            """
            Partitioning by greedily maximizing the graph modularity
            """
            communities = nx.community.greedy_modularity_communities(self, resolution=resolution)
            clusters = [0 for _ in range(len(self.nodes()))]

            for i, community in enumerate(communities):
                for j, node in enumerate(community):
                    clusters[node] = i

        elif method == "louvain":
            """
            Partitioning using the louvain algorithm
            """

            communities = nx.community.louvain_communities(self, resolution=resolution)
            clusters = [0 for _ in range(len(self.nodes()))]

            for i, community in enumerate(communities):
                for j, node in enumerate(community):
                    clusters[list(self.nodes()).index(node)] = i

        else:
            raise Exception("Not a valid partitioning method (spectral, greedy, louvain[preferred])")

        return nx.algorithms.community.modularity(self, communities=communities, resolution=resolution), clusters
