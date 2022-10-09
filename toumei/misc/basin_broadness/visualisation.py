import dash
import torch
from dash import Input, Output, dcc, html
import dash_bootstrap_components as dbc
import networkx as nx
import plotly.graph_objs as go
from colour import Color
from datetime import datetime
import json
import plotly.express as px

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
    pos = nx.layout.multipartite_layout(graph, subset_key="layer", scale=4)

    for node in graph.nodes:
        graph.nodes[node]['pos'] = list(pos[node])

    traces = []

    node_trace = go.Scatter(x=[], y=[], hovertext=[], text=[], mode='markers+text', textposition="bottom center",
                            hoverinfo="text", marker={'size': 50, 'color': [], 'symbol': 'circle-dot',
                                                      'line_width': 2, 'line_color': 'midnightblue', 'showscale': True,
                                                      'colorscale': 'Sunset',
                                                      'colorbar': dict(
                                                          thickness=15,
                                                          title='||Eigenvalue||',
                                                          xanchor='left',
                                                          titleside='right')
                                                      },
                            selected={
                                'marker':
                                    {
                                        'color': 'DeepSkyBlue'
                                    }
                            }

                            )

    negative_colors = "(255, 8, 51)"
    positive_colors = "(0, 224, 145)"

    index = 0
    for edge in graph.edges:
        x0, y0 = graph.nodes[edge[0]]['pos']
        x1, y1 = graph.nodes[edge[1]]['pos']
        weight = graph.edges[edge]['weight']
        value = graph.edges[edge]['value']
        trace = go.Scatter(x=tuple([x0, x1]), y=tuple([y0, y1]),
                           mode='lines',
                           line={'width': 2},
                           marker=dict(color='rgb' + negative_colors if value < 0 else 'rgb' + positive_colors),
                           line_shape='spline',
                           opacity=1)
        weight_range = weight/100
        weight_band = go.Scatter(x=tuple([x0 , x1, x1, x0]), y=tuple([y0-weight_range, y1-weight_range, y1+weight_range, y0+weight_range]),
                                 fill='toself',
                                 fillcolor=('rgba' + negative_colors[:-1] if value < 0 else 'rgba' + positive_colors[:-1]) + ',0.2)',
                                 line=dict(color='rgba(255,255,255,0)'),
                                 hoverinfo="skip",
                                 showlegend=False
                                 )
        traces.append(trace)
        traces.append(weight_band)
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
    colors = []
    for node in graph.nodes():
        x, y = graph.nodes[node]['pos']
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_trace['text'] += tuple([node])
        layer = int(node[5]) - 1
        unit = int(node[-1])
        try:
            _, L, _ = w[layer].orthogonal_basis
            L[torch.logical_and(L >= 0, L <= 1e-4)] = 0
            L[torch.logical_and(L <= 0, L >= -1e-4)] = 0
            colors.append(torch.sqrt(L[unit]).item())
            node_trace['hovertext'] += tuple([torch.sqrt(L[unit]).item()])
        except:
            colors.append('DeepPink')
        index = index + 1

    node_trace.marker.color = colors
    traces.append(node_trace)

    figure = {
        "data": traces,
        "layout": go.Layout(title='Network Visualisation', showlegend=False, hovermode='closest',
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

accordion = html.Div(
    dbc.Accordion(
        [
            dbc.AccordionItem(
                "This is the content of the first section", title="What is this?"
            ),
            dbc.AccordionItem(
                "This is the content of the second section", title="How does it work?"
            ),
            dbc.AccordionItem(
                "This is the content of the third section", title="I want to learn more"
            ),
        ],
        start_collapsed=True,
    ),
    style={"margin-top": "2rem"}
)

sidebar = html.Div(
    [

        html.H2([
            html.I(className="bi bi-layers-half", style={"padding-right": "10px"}),
            "Feature Orthogonalisation"
        ], className="display-4", style={"font-size": "30px"}),
        html.Hr(),
        html.P("Orthogonalise models using the hilbert space formalism.", className="lead"),
        accordion

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
            dbc.Spinner(html.Div(
                id="card-content",
                children=[dcc.Graph(id="my-graph",
                                    figure=build_figure(model))],
            ), color="primary"),
        ]),
    ],
    style=PLOT_STYLE
)

card_content = [
    dbc.CardHeader([html.I(className="bi bi-lightbulb", style={"padding-right": "10px"}), "Context info"]),
    dbc.CardBody(
        [
            dbc.Alert([
                html.H5([html.I(className="bi bi-grid-3x3", style={"padding-right": "10px"}), "Activation Matrix"],
                        className="card-title"),
                html.P(
                    "The L2 inner-product matrix of the neurons for the selected layer"
                ),
                dbc.Spinner(html.Div(id="activation-map")),
            ], color='primary'),
            dbc.Alert([
                html.H5([html.I(className="bi bi-123", style={"padding-right": "10px"}), "Eigenvalues"],
                        className="card-title"),
                html.P(
                    "The eigenvalues of the l2 inner product matrix",
                    className="card-text",
                ),
                dbc.Spinner(html.Div(id="eigenvalues")),
            ], color='secondary')
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
        content = dcc.Graph(id="my-graph", figure=build_figure(model), mathjax=True)
    else:
        content = dcc.Graph(id="my-graph", figure=build_figure(ortho_model), mathjax=True)
    return content

@app.callback(
    Output('activation-map', 'children'),
    [Input('my-graph', 'clickData')])
def display_activation_map(clickData):
    if clickData is None:
        return ""
    text = clickData['points'][0]['text']
    layer = int(text[5]) - 1
    return dcc.Graph(figure=px.imshow(w[layer].activation_matrix.detach().numpy()), style={"height": "300px"})

@app.callback(
    Output('eigenvalues', 'children'),
    [Input('my-graph', 'clickData')])
def display_eigenvalues(clickData):
    if clickData is None:
        return ""
    text = clickData['points'][0]['text']
    layer = int(text[5]) - 1
    _, L, _ = w[layer].orthogonal_basis
    fig = go.Figure([go.Bar(x=[i for i in range(L.shape[0])], y=torch.sqrt(L).view(-1).detach().numpy())])
    return dcc.Graph(figure=fig, style={"height": "285px"})


if __name__ == '__main__':
    app.run_server(debug=True)
