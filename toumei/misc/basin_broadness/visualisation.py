import dash
import torch
from dash import Input, Output, dcc, html
import dash_bootstrap_components as dbc
import networkx as nx
import plotly.graph_objs as go
import pandas as pd
from colour import Color
from datetime import datetime
from textwrap import dedent as d
import json

from torch import nn

from toumei.mlp import MLPGraph
from toumei.models import SimpleMLP

model = SimpleMLP(2, 4, 1, activation=nn.Sigmoid())
model.load_state_dict(torch.load("xor_model_2_4_1.pth"))

graph = MLPGraph(model)
pos = nx.layout.multipartite_layout(graph, subset_key="layer", scale=2)

for node in graph.nodes:
    graph.nodes[node]['pos'] = list(pos[node])

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Feature Orthogonalisation"

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "18rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

sidebar = html.Div(
    [
        html.H2("Feature Orthogonalisation", className="display-4", style={"font-size": "30px"}),
        html.Hr(),
        html.P(
            "Orthogonalise models using the hilbert space formalism", className="lead"
        )
    ],
    style=SIDEBAR_STYLE,
)

CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}


content = html.Div([])

app.layout = html.Div([sidebar, content])

if __name__ == '__main__':
    app.run_server(debug=True)
