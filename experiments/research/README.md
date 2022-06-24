# Investigating (randomly) modulary varying goals in modern deep learning architectures

This research project aims to evaluate the impact of randomly modulary varying goals on model modularity and to give us an idea of the problem. It was proposed by <a href="https://www.lesswrong.com/posts/99WtcMpsRqZcrocCd/ten-experiments-in-modularity-which-we-d-like-you-to-run">this post</a>.

We implemented the network modularity metric by performing model graphification using weights (MLPs) / cnn-kernel matrix-slice norms (CNNs) and then spectral clustering (see [model_graph.py](https://github.com/LuanAdemi/toumei/blob/master/toumei/misc/model_graph.py)) the resulting graph as proposed in [1] and [2].

### Building a baseline model with high functional modularity

In order to test our implementation of the modularity metric and to have a baseline model we can compare other models to, we started by engineering a model with a high functional modularity.
We trained two sperate MLPs to perform different subtask of the task *"recognise two MNIST numbers and add them"*.

### Measuring the modularity of the baseline model 

### Comparing the modularity to a model just trained on the whole task

### Training a model using modulary varying goals

### Comparing the modularity to the baseline model

# References
- [1] [Quantifying local specialization in Deep Neural Networks](https://arxiv.org/pdf/2110.08058.pdf)
- [2] [Spontaneous evolution of modularity and network motifs](https://www.pnas.org/doi/pdf/10.1073/pnas.0503610102)