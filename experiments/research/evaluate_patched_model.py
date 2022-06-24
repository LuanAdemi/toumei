import os
import sys

import torch

import matplotlib.pyplot as plt

from experiments.research.patch_model import PatchedModel
from toumei.misc import MLPGraph, CNNGraph
from toumei.models import SimpleMLP, SimpleCNN

import networkx as nx

k = 8

device = torch.device("cuda")

network = PatchedModel().to(device)
network.load_state_dict(torch.load("models/patched_model_param.pth"))

m_1 = SimpleCNN(2, 1)
m_1.load_state_dict(torch.load("models/mvg.pth"))

graph1 = MLPGraph(network)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
fig.suptitle(f"Spectral Clustering k={k} - Q={graph1.get_model_modularity(n_clusters=k)}")

communities, clusters = graph1._spectral_clustering(n_clusters=k)

ax1.set_title('Multipartite Layout')
nx.draw_networkx_nodes(graph1, pos=nx.multipartite_layout(graph1, subset_key="layer", scale=2),
                       cmap=plt.get_cmap('rainbow'), node_color=clusters, node_size=30, ax=ax1)

ax2.set_title('Spring Layout')
nx.draw_networkx_nodes(graph1, pos=nx.spring_layout(graph1), cmap=plt.get_cmap('rainbow'),
                       node_color=clusters, node_size=30, ax=ax2)

plt.show()
