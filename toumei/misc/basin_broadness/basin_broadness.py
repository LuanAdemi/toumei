import torch
from pyvis.network import Network
from torch import nn

from toumei.mlp import MLPGraph
from toumei.models import SimpleMLP
from base import MLPWrapper

if __name__ == '__main__':
    inputs = torch.Tensor([
        [0., 0.],
        [0., 1.],
        [1., 0.],
        [1., 1.]])

    labels = torch.Tensor([0., 1., 1., 0.]).reshape(inputs.shape[0], 1)

    model = SimpleMLP(2, 3, 1, activation=nn.Sigmoid())
    model.load_state_dict(torch.load("xor00.pth"))
    w = MLPWrapper(model, inputs, labels)

    ortho_model = w.orthogonal_model(act=nn.Sigmoid())
    print(ortho_model)

    graph1 = MLPGraph(model=ortho_model)
    graph2 = MLPGraph(model=model)

    module1, L1 = w[0].orthogonalise()

    inp = torch.cat([inputs, torch.ones(4).unsqueeze(1)], dim=1)

    print(L1, ortho_model(inp))

    nt = Network('900px', '1900px')
    nt.from_nx(graph1)
    nt.set_options("""
            const options = {
      "nodes": {
        "borderWidth": null,
        "borderWidthSelected": null,
        "opacity": null,
        "size": null
      },
      "edges": {
        "color": {
          "inherit": true
        },
        "selfReferenceSize": null,
        "selfReference": {
          "angle": 0.7853981633974483
        },
        "smooth": false
      },
      "layout": {
        "hierarchical": {
          "enabled": true,
          "direction": "LR",
          "sortMethod": "directed"
        }
      },
      "interaction": {
        "hover": true,
        "keyboard": {
          "enabled": true
        },
        "multiselect": true,
        "navigationButtons": true
      },
      "manipulation": {
        "enabled": true
      },
      "physics": {
        "hierarchicalRepulsion": {
          "centralGravity": 0,
          "avoidOverlap": null
        },
        "minVelocity": 0.75,
        "solver": "hierarchicalRepulsion"
      }
    }
        """)
    nt.show('nx.html')