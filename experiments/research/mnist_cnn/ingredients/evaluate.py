from sacred import Ingredient

evaluate_ingredient = Ingredient('evaluate')
from toumei.misc import CNNGraph, MLPGraph
from utils import *



@evaluate_ingredient.config
def cfg():
    modularity = True
    partitioning_method = 'louvain'
    feature_vis = False
    ntk_plots = False


@evaluate_ingredient.capture
def evaluate_model(_run, modularity, partitioning_method, feature_vis, ntk_plots, model, save_path):
    if modularity:
        cnn_graph = CNNGraph(model)
        mlp_graph = MLPGraph(model)

        Q_cnn = calc_modularity(cnn_graph, save_path + "cnn_modularity_plot.png", partitioning_method)
        Q_mlp = calc_modularity(mlp_graph, save_path + "mlp_modularity_plot.png", partitioning_method)

        _run.log_scalar("Q_CNN", Q_cnn)
        _run.log_scalar("Q_MLP", Q_mlp)

        _run.add_artifact(save_path + "cnn_modularity_plot.png", "cnn_modularity_plot.png", content_type="image/png")
        _run.add_artifact(save_path + "mlp_modularity_plot.png", "mlp_modularity_plot.png", content_type="image/png")

    if feature_vis:
        """
        TODO
        """

    if ntk_plots:
        """
        TODO
        """