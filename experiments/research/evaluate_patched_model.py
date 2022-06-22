import os
import sys

import torch

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'

sys.path.append("../../")
from experiments.research.patch_model import PatchedModel
from toumei.misc import MLPGraph

device = torch.device("cuda")

network = PatchedModel().to(device)
network.load_state_dict(torch.load("patched_model.pth"))

graph = MLPGraph(network)
print(graph.get_model_modularity())
