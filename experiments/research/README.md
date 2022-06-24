# Investigating (randomly) modulary varying goals in modern deep learning architectures

> *This is a rough outline on what we have tried so far.*

This research project aims to evaluate the impact of randomly modulary varying goals on model modularity and to give us an idea of the problem. It was proposed by <a href="https://www.lesswrong.com/posts/99WtcMpsRqZcrocCd/ten-experiments-in-modularity-which-we-d-like-you-to-run">this post</a>.

We implemented the network modularity metric by performing model graphification using weights (MLPs) / cnn-kernel matrix-slice norms (CNNs) and then spectral clustering (see [model_graph.py](https://github.com/LuanAdemi/toumei/blob/master/toumei/misc/model_graph.py)) the resulting graph as proposed in [1] and [2].

## Building a baseline model with high functional modularity

In order to test our implementation of the modularity metric and to have a baseline model we can compare other models to, we started by engineering a model with a high functional modularity.

Two sperate MLPs $F$ and $G$ were trained to perform different subtask of the task *"recognise two MNIST numbers and add them"* and patched to a single fully connected model $H := G(C_{F_1,F_2})$ with $C$ beeing a model stacking $F_1$ and $F_2$ with cross connections added.

> *See [patch_model.py](https://github.com/LuanAdemi/toumei/blob/master/experiments/research/patch_model.py)*

After patching the model and adding the corresponding cross connections, we trained the model on the task for a few episodes with (partially) frozen paramaters (one time freezing the whole module including and one time excluding the cross connections).

The resulting model has a above average Q-Value and can therefore be called "modular" according to the metric, altough we expected this number to be way higher.

### Frozen parameters including cross connections

![](https://raw.githubusercontent.com/LuanAdemi/toumei/master/experiments/research/plots/patched_mlp.png)

### Frozen parameters excluding cross connections

![](https://raw.githubusercontent.com/LuanAdemi/toumei/master/experiments/research/plots/patched_mlp_param.png)


### Comparing the modularity to a model just trained on the whole task

Comparing the patched model to a model just trained on the task directly puts this number in perspective

![](https://raw.githubusercontent.com/LuanAdemi/toumei/master/experiments/research/plots/mlp.png)

### Training a model using modulary varying goals


# References
- [1] [Quantifying local specialization in Deep Neural Networks](https://arxiv.org/pdf/2110.08058.pdf)
- [2] [Spontaneous evolution of modularity and network motifs](https://www.pnas.org/doi/pdf/10.1073/pnas.0503610102)
