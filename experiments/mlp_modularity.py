from toumei.misc import MLPGraph
from toumei.models import SimpleMLP

from toumei.cnns.objectives.utils import set_seed

"""
A script showcasing the modularity measurement of a simple MLP 
using the pipeline introduced in https://arxiv.org/pdf/2110.08058.pdf
"""

set_seed(42)

# create a simple mlp
model = SimpleMLP(4, 4)

# create graph from the model
graph = MLPGraph(model)

# calculate the modularity
print(graph.get_model_modularity())
