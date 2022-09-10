import sys

from toumei.models import SimpleMLP
from toumei.mlp import MLPGraph
from toumei.cnns.featurevis.objectives.utils import set_seed


def test_mlp_modularity():
    set_seed(42)

    # create a simple mlp
    model = SimpleMLP(4, 4)

    # create graph from the model
    graph = MLPGraph(model)

    # calculate the modularity
    assert graph.get_model_modularity() == (0.15438260497329004, [0, 0, 0, 0, 0, 1, 2, 3])
