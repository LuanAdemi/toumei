import networkx as nx

import toumei.cnns.featurevis.objectives as obj
import toumei.cnns.featurevis.parameterization as param

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from toumei.general import DatasetFinder

import matplotlib.gridspec as gridspec


import torchvision.transforms as T

from mpl_toolkits.axes_grid1 import ImageGrid

# compose the image transformation for regularization through transformations robustness
transform = T.Compose([
    T.Pad(12),
    T.RandomRotation((-10, 11)),
])

device = torch.device('cuda')


class LinearTail(nn.Module):
    def __init__(self, model):
        super(LinearTail, self).__init__()
        super().__init__()
        self.linear_module = nn.Sequential(
            model.fc1,
            model.relu_fc1,
            model.fc2,
            model.relu_fc2,
            model.fc3,
            model.relu_fc3,
            model.fc4
        )

    def forward(self, x):
        x = self.linear_module(x)
        return x


def calc_modularity(graph, f, method='louvain'):
    Q, clusters = graph.get_model_modularity(method=method)

    community_to_color = {
        0: 'tab:blue',
        1: 'tab:orange',
        2: 'tab:green',
        3: 'tab:red',
        4: 'blue',
        5: 'orange',
        6: 'red',
        7: 'green',
        8: 'yellow',
        9: 'brown',
        10: 'pink',
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle(f"Louvain Partitioning - Q={Q}")

    community = {list(graph.nodes())[n]: c for n, c in enumerate(clusters)}
    node_color = {node: community_to_color[community_id] for node, community_id in community.items()}

    ax1.set_title('Multipartite Layout')
    nx.draw(graph, pos=nx.multipartite_layout(graph, subset_key="layer", scale=2),
            node_color=list(node_color.values()), node_size=100, ax=ax1,
            font_size=8, edge_color='lightgray', horizontalalignment='left', verticalalignment='bottom')

    ax2.set_title('Spring Layout')
    nx.draw(graph, pos=nx.spring_layout(graph),
            node_color=list(node_color.values()), node_size=100, ax=ax2,
            font_size=8, edge_color='lightgray', horizontalalignment='left', verticalalignment='bottom')

    plt.savefig(f)

    return Q


def visualise_objective(model, objective):
    fv = obj.Pipeline(
        # the image generator object
        param.Transform(param.FFTImage(1, 2, 224, 224), transform),

        # the objective function
        objective
    )

    # attach the pipeline to the model
    fv.attach(model)

    # send the objective to the gpu
    fv.to(device)

    # optimize the objective
    fv.optimize(verbose=False)

    # get the results
    return fv.generator.numpy()


def visualise_dataset_samples(model, dataset, objective, sample_size, k):
    df = DatasetFinder(dataset, objective, sample_size)
    df.attach(model)
    df.optimize(verbose=False)

    return df.get_topk(k)


def visualise_unit(model, unit, dataset, k):
    if "conv" in unit:
        objective = obj.Channel
    else:
        objective = obj.Neuron

    sample_result = visualise_dataset_samples(model, dataset, objective(unit), sample_size=1024, k=k)
    fv_results = visualise_objective(model, objective(unit))
    return sample_result, fv_results


def plot_unit(model, unit, dataset, k, save_plot=False):
    s, fv = visualise_unit(model, unit, dataset, k)

    fig = plt.figure(figsize=(8, 6), constrained_layout=True)
    fig.suptitle(f"Feature Visualisation - {unit}")

    gs = gridspec.GridSpec(k // 2, k // 2 + 2, figure=fig)

    ax = plt.subplot(gs[:k//4, :k//4])
    ax.axis("off")
    ax.imshow(fv[:, :, 0], cmap="gray", aspect='auto')

    ax = plt.subplot(gs[k//4:, :k//4])
    ax.axis("off")
    ax.imshow(fv[:, :, 1], cmap="gray", aspect='auto')

    row_idx = 0
    col_idx = 1
    for i in range(2):
        for j in range(k):
            if col_idx > k//2:
                col_idx = 2
                row_idx += 1
            else:
                col_idx += 1

            ax = plt.subplot(gs[row_idx, col_idx])
            ax.axis("off")
            ax.imshow(s[j][:, :, i], cmap="gray", aspect='auto')

    if save_plot:
        plt.savefig(f"plots/{unit}.png")
    else:
        plt.show()





def plot_job(job, job_results):
    module, instances, type = job

    fig = plt.figure(figsize=(10., 10.))
    fig.suptitle(f"Feature Visualisation - {module}")
    grid1 = ImageGrid(fig, 211,  # similar to subplot(111)
                      nrows_ncols=(max(instances // 8, 1), 8),
                      axes_pad=0.1,  # pad between axes in inch.
                      )

    grid2 = ImageGrid(fig, 212,  # similar to subplot(111)
                      nrows_ncols=(max(instances // 8, 1), 8),
                      axes_pad=0.1,  # pad between axes in inch.
                      )

    for ax1, ax2, im in zip(grid1, grid2, job_results):
        # Iterating over the grid returns the Axes.
        ax1.imshow(im[:, :, 0], cmap="gray")
        ax2.imshow(im[:, :, 1], cmap="gray")

    plt.show()
