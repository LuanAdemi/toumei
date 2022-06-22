import sys

sys.path.append("../")
from toumei.models import SimpleMLP
from toumei.misc import MLPGraph
from toumei.cnns.objectives.utils import set_seed


def test_mlp_modularity():
    set_seed(42)

    # create a simple mlp
    model = SimpleMLP(4, 4)

    # create graph from the model
    graph = MLPGraph(model)

    # calculate the modularity
    assert graph.get_model_modularity() == -0.0657500117616066
