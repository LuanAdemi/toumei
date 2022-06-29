import torch

import matplotlib.pyplot as plt

from toumei.misc import MLPGraph
from toumei.models import SimpleMLP

import networkx as nx

k = 6

device = torch.device("cuda")

network = SimpleMLP(2 * 28 * 28, 1024, 128, 64, 32, 1).to(device)
network.load_state_dict(torch.load("modular_varying_goals_model_1.pth"))

graph1 = MLPGraph(network)

Q = graph1.get_model_modularity(n_clusters=k)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
fig.suptitle(f"Spectral Clustering k={k} - Q={Q}")

communities, clusters = graph1._spectral_clustering(n_clusters=k)
ax1.set_title('Multipartite Layout')
nx.draw_networkx_nodes(graph1, pos=nx.multipartite_layout(graph1, subset_key="layer", scale=2),
                       cmap=plt.get_cmap('rainbow'), node_color=clusters, node_size=30, ax=ax1)

ax2.set_title('Spring Layout')
nx.draw_networkx_nodes(graph1, pos=nx.spring_layout(graph1), cmap=plt.get_cmap('rainbow'),
                       node_color=clusters, node_size=30, ax=ax2)

plt.show()
