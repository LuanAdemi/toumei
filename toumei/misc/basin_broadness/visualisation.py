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
from base import MLPWrapper


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


def build_figure(model):
    graph = MLPGraph(model)
    pos = nx.layout.multipartite_layout(graph, subset_key="layer")

    for node in graph.nodes:
        graph.nodes[node]['pos'] = list(pos[node])

    traces = []

    node_trace = go.Scatter(x=[], y=[], hovertext=[], text=[], mode='markers+text', textposition="bottom center",
                            hoverinfo="text", marker={'size': 50, 'color': 'LightSkyBlue'})

    negative_colors = list(Color('lightcoral').range_to(Color('darkred'), len(graph.edges())))
    positive_colors = list(Color('greenyellow').range_to(Color('limegreen'), len(graph.edges())))
    negative_colors = ['rgb' + str(x.rgb) for x in negative_colors]
    positive_colors = ['rgb' + str(x.rgb) for x in positive_colors]

    index = 0
    for edge in graph.edges:
        x0, y0 = graph.nodes[edge[0]]['pos']
        x1, y1 = graph.nodes[edge[1]]['pos']
        weight = graph.edges[edge]['weight']
        value = graph.edges[edge]['value']
        trace = go.Scatter(x=tuple([x0, x1, None]), y=tuple([y0, y1, None]),
                           mode='lines',
                           line={'width': weight},
                           marker=dict(color=negative_colors[index] if value < 0 else positive_colors[index]),
                           line_shape='spline',
                           opacity=1)
        traces.append(trace)
        index = index + 1

    middle_hover_trace = go.Scatter(x=[], y=[], hovertext=[], mode='markers', hoverinfo="text",
                                    marker={'size': 20, 'color': 'LightSkyBlue'}, opacity=0)
    index = 0
    for edge in graph.edges:
        x0, y0 = graph.nodes[edge[0]]['pos']
        x1, y1 = graph.nodes[edge[1]]['pos']
        hovertext = str(graph.edges[edge]['value'])
        middle_hover_trace['x'] += tuple([(x0 + x1) / 2])
        middle_hover_trace['y'] += tuple([(y0 + y1) / 2])
        middle_hover_trace['hovertext'] += tuple([hovertext])
        index = index + 1

    traces.append(middle_hover_trace)

    index = 0
    for node in graph.nodes():
        x, y = graph.nodes[node]['pos']
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_trace['hovertext'] += tuple([node])
        index = index + 1

    traces.append(node_trace)

    figure = {
        "data": traces,
        "layout": go.Layout(title='Interactive Transaction Visualization', showlegend=False, hovermode='closest',
                            margin={'b': 40, 'l': 40, 'r': 40, 't': 40},
                            xaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                            yaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                            height=845,
                            clickmode='event+select'
                            )}

    return figure


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP])
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

        html.H2([
            html.I(className="bi bi-layers-half", style={"padding-right": "10px"}),
            "Feature Orthogonalisation"
        ], className="display-4", style={"font-size": "30px"}),
        html.Hr(),
        html.P("Orthogonalise models using the hilbert space formalism.", className="lead")
    ],
    style=SIDEBAR_STYLE,
)

PLOT_STYLE = {
    "margin-left": "20rem",
    "margin-right": "0rem",
    "margin-top": "1rem",
    "margin-bottom": "1rem"
}

CONTEXT_STYLE = {
    "margin-top": "1rem",
    "width": "28rem",
    "margin-left": "1rem"
}


plot = dbc.Card(
    [
        dbc.CardHeader(
            dbc.Tabs(
                [
                    dbc.Tab(label="Original model", tab_id="original"),
                    dbc.Tab(label="Orthogonal model", tab_id="orthogonal"),
                ],
                id="card-tabs",
                active_tab="original",
            )
        ),
        dbc.CardBody([
            html.Div(
                id="card-content",
                children=[dcc.Graph(id="my-graph",
                                    figure=build_figure(model))],
            ),
        ]),
    ],
    style=PLOT_STYLE
)

card_content = [
    dbc.CardHeader([html.I(className="bi bi-lightbulb", style={"padding-right": "10px"}), "Context info"]),
    dbc.CardBody(
        [
            html.H5([html.I(className="bi bi-grid-3x3", style={"padding-right": "10px"}), "Activation Matrix"],
                    className="card-title"),
            html.P(
                "The L2 inner-product matrix of the neurons for the selected layer",
                className="card-text",
            ),
            html.H5([html.I(className="bi bi-123", style={"padding-right": "10px"}), "Eigenvalues"],
                    className="card-title"),
            html.P(
                "This is some card content that we'll reuse",
                className="card-text",
            ),
        ]
    ),
]

context_info = dbc.Row(
    [
        dbc.Col(dbc.Card(card_content, color="dark", outline=True)),
    ], style=CONTEXT_STYLE)


app.layout = html.Div([sidebar,
                       dbc.Row([
                           dbc.Col(plot, style={"padding": "0"}),
                           dbc.Col(context_info, width=3, style={"padding": "0"})])
                       ])


@app.callback(
    Output("card-content", "children"), [Input("card-tabs", "active_tab")]
)
def tab_content(active_tab):
    if active_tab == "original":
        content = dcc.Graph(id="my-graph", figure=build_figure(model))
    else:
        content = dcc.Graph(id="my-graph", figure=build_figure(ortho_model))
    return content


if __name__ == '__main__':
    app.run_server(debug=True)
