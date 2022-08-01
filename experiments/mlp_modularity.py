from toumei.mlp import MLPGraph
from toumei.models import SimpleMLP

model = SimpleMLP(4, 4)
graph = MLPGraph(model)

print(graph.get_model_modularity())
