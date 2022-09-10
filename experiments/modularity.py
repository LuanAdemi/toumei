import torch

from toumei.mlp import MLPGraph
from toumei.cnns import CNNGraph
from toumei.models import SimpleMLP, SimpleCNN

from toumei.cnns.featurevis.objectives.utils import set_seed

"""
A script showcasing the modularity measurement of a simple MLP 
using the pipeline introduced in https://arxiv.org/pdf/2110.08058.pdf
"""

# create the models
mlp = SimpleMLP(4, 4)
cnn = SimpleCNN(1, 10)

# create graph from the model
mlp_graph = MLPGraph(mlp)
cnn_graph = CNNGraph(cnn)

# calculate the modularity
print(mlp_graph.get_model_modularity())
print(cnn_graph.get_model_modularity())
