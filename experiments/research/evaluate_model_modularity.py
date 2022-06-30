import matplotlib.pyplot as plt
import networkx as nx
import torch
from netgraph import Graph

from toumei.misc import MLPGraph
from toumei.models import SimpleMLP

k = 6

device = torch.device("cuda")

network = SimpleMLP(16, 32, 64, 32, 16, 1).to(device)
network.load_state_dict(torch.load("onehot_addition/models/binary_addition_model_big.pth"))

graph1 = MLPGraph(network)

Q, clusters = graph1.get_model_modularity(n_clusters=k, method="louvain")

community_to_color = {
    0: 'tab:blue',
    1: 'tab:orange',
    2: 'tab:green',
    3: 'tab:red',
    4 : 'blue',
    5 : 'orange'
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
fig.suptitle(f"MVG - OneHot - Louvain - Q={Q}")

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
