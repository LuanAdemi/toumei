import torch
import sys
sys.path.append("../../")
from toumei.models import SimpleMLP, SimpleCNN
from toumei.misc import CNNGraph

device = torch.device("cuda")

network = SimpleCNN(1, 10).to(device)
network.load_state_dict(torch.load("mnist_model.pth"))

network.eval()
graph = CNNGraph(network)
print(graph.get_model_modularity())