import sys

import torch

sys.path.append("../../")
from toumei.models import SimpleCNN
from toumei.misc import CNNGraph

device = torch.device("cuda")

network = SimpleCNN(2, 1).to(device)
network.load_state_dict(torch.load("model.pth"))

network.eval()
graph = CNNGraph(network)
print(graph.get_model_modularity())
