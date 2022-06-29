import matplotlib.pyplot as plt
import networkx as nx
import torch

from toumei.misc import MLPGraph
from toumei.models import SimpleMLP

from netgraph import Graph

k = 4

device = torch.device("cuda")

network = SimpleMLP(16, 8, 4, 1).to(device)
network.load_state_dict(torch.load("binary_addition/binary_addition_model_no_mvg.pth"))

graph1 = MLPGraph(network)

Q, clusters = graph1.get_model_modularity(n_clusters=k, method="spectral")

community_to_color = {
    0 : 'tab:blue',
    1 : 'tab:orange',
    2 : 'tab:green',
    3 : 'tab:red',
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
fig.suptitle(f"No MVG - Spectral Clustering k=4 - Q={Q}")

community = {list(graph1.nodes())[n]: c for n, c in enumerate(clusters)}
node_color = {node: community_to_color[community_id] for node, community_id in community.items()}

ax1.set_title('Multipartite Layout')
nx.draw_networkx(graph1, pos=nx.multipartite_layout(graph1, subset_key="layer", scale=2),
                 node_color=list(node_color.values()), node_size=100, ax=ax1,
                 font_size=8, edge_color='lightgray', horizontalalignment='left', verticalalignment='bottom')

graph1.__class__ = nx.Graph
Graph(graph1,
      node_color=node_color, node_edge_width=0, edge_alpha=0.1,
      node_layout='community', node_layout_kwargs=dict(node_to_community=community),
      edge_layout='bundled', edge_layout_kwargs=dict(k=2000), ax=ax2, node_labels=True, node_label_fontdict=dict(size=10))

plt.show()
