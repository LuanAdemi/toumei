from collections.abc import Iterator
from copy import deepcopy

import torch
import torch.nn as nn
from torch import autograd
from torch.nn import Parameter

from toumei.models import SimpleMLP
from toumei.probe import print_modules

from toumei.mlp.mlp_graph import MLPGraph

from pyvis.network import Network


class SourceNode(nn.Module):
    """
    This will be the head node of our linked list, which exposes the dataset to the next nodes
    """
    def __init__(self, data):
        super(SourceNode, self).__init__()

        self.data = data

    def forward(self, x):
        """
        Returns the dataset

        :param x: the input tensor
        :return: the hidden state/output
        """
        return self.data

    def forward_pass(self):
        """
        Performs a forward pass by making recursive calls to all previous nodes

        :return: the hidden state of this node
        """
        # this is the tail node, so no preceding nodes exist
        return self.data


class LinearNode(nn.Module):
    """
    Wraps a linear layer and adds functionality to it
    """
    def __init__(self, name: str, parent: nn.Module, child: nn.Module, prv: nn.Module = None):
        super(LinearNode, self).__init__()

        # the node name (the name of the wrapped module)
        self.name = name

        # the parent container
        self.parent = parent

        # the wrapped linear layer
        self.child = child

        # the preceding node
        self.prev = prv

        # vectorized params
        self.params = nn.utils.parameters_to_vector(self.parameters())

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return self.child.parameters()

    def forward(self, x):
        """
        Forwards a tensor through the node

        :param x: the input tensor
        :return: the hidden state/output
        """
        return self.child.forward(x)

    def forward_pass(self):
        """
        Performs a forward pass by making recursive calls to all previous nodes

        :return: the hidden state of this node
        """
        return self.forward(self.prev.forward_pass())

    @property
    def activation_matrix(self):
        activations = self.prev.forward_pass()

        result = torch.zeros(size=(activations.shape[-1], activations.shape[-1]))

        for a in activations:
            result += torch.outer(a, a.T)

        return result

    @property
    def orthogonal_basis(self):
        """
        Returns the orthogonal (eigen) basis of the parameter space

        :return: the orthogonal basis
        """
        L, V = torch.linalg.eig(self.activation_matrix)
        return V.real, torch.diag(L)


class MLPWrapper(nn.Module):
    """
    Implements a linked array list like structure.

    Nodes (LinearLayers) are stored in a single linked array list and can be accessed dynamically.

    A node has access to every preceding node, so it can dynamically build the computation graph.
    """
    def __init__(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor, loss_func=nn.MSELoss()):
        super(MLPWrapper, self).__init__()

        # the dataset and loss function
        self.X = x
        self.Y = y
        self.loss_func = loss_func

        # the model
        self.model = model

        # a container for the single linked array list
        self.nodes = []

        # vectorized model parameters
        self.params = nn.utils.parameters_to_vector(self.model.parameters())

        # build the linked array list
        prev = SourceNode(data=self.X)

        for key, value in self.model.named_modules():
            if isinstance(value, nn.Linear) or isinstance(value, DummyLayer):
                node = LinearNode(key, self, value, prev)
                self.nodes.append(node)
                prev = node

        # a dictionary mapping the node names to an integer index
        self.key_to_idx = {n.name: i for i, n in enumerate(self.nodes)}

    def forward(self):
        """
        Makes a recursive call with the last element in the linked list of the nodes

        :return: the model output
        """
        return self.nodes[-1].forward_pass()

    def forward_pass(self):
        """
        Performs a forward pass

        :return: the model output and the loss
        """
        out = self.forward()
        loss = self.loss_func(out, self.Y)
        return out, loss

    def intercepted_forward_pass(self, node):
        """
        Intercepts the forward pass at the specified node and extracts the hidden state

        :param node: the node where the forward pass should be intercepted
        :return: the hidden state and the loss
        """

        # retrieve the hidden state of the specified node
        n_out = node.forward_pass()

        out = n_out

        # continue the forward pass
        for node in self.nodes[self.nodes.index(node)+1:]:
            out = node(out)

        loss = self.loss_func(out, self.Y)
        return n_out, loss

    def orthogonal_model(self, inplace=False):
        """
        This is the main algorithm.
        It collects the orthogonal features of each node (layer) and builds the corresponding orthogonal model.

        :param log_mask: If true, the parameters will be masked according to their absolute log values
        :param inplace  if true, the algorithm will be performed on the passed model, else it will work on a copy
        :return: the orthogonal model
        """

        if inplace:
            ortho_model = model
        else:
            ortho_model = deepcopy(self.model)

        current_node = 0

        # iterate over each module of the orthogonal model
        for name, module in ortho_model.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, DummyLayer):
                # set the parameters to the orthogonal ones
                ortho_param = self[current_node].orthogonal_parameters

                nn.utils.vector_to_parameters(ortho_param, module.parameters())

                current_node += 1



        return ortho_model

    def __getitem__(self, item):
        """
        Implements two types of indexing nodes.
        1) by index
        2) by name

        :param item: an integer index or the node name
        :return: the corresponding node
        """
        if isinstance(item, str):
            return self.nodes[self.key_to_idx[item]]
        else:
            return self.nodes[item]


class DummyLayer(nn.Module):
    def __init__(self):
        super(DummyLayer, self).__init__()
        self.params = nn.Parameter(torch.ones((3,), dtype=torch.float))

    def forward(self, x):
        return self.params[0] * x + self.params[1] * x + self.params[2] * x


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()

        self.dl = DummyLayer()

    def forward(self, x):
        return self.dl(x)


if __name__ == '__main__':
    model = SimpleMLP(1, 4, 8, 3, 1)
    inputs = torch.randn(size=(1024, 1), dtype=torch.float)
    labels = model(inputs)
    w = MLPWrapper(model, inputs, labels)


    print(w[1].orthogonal_basis)




