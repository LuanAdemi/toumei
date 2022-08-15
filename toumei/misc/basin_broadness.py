from collections.abc import Iterator
from copy import deepcopy

import torch
import torch.nn as nn
from torch import autograd
from torch.nn import Parameter

from toumei.models import SimpleMLP
from toumei.probe import print_modules


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
        if self.prev is not None:
            return self.forward(self.prev.forward_pass())
        else:
            # this is the tail node, so no preceding nodes exist
            return self.forward(self.parent.X)

    @property
    def d2l_d2f(self):
        """
        Calculates the first term of the hessian decomposition.

        :return: the corresponding gradient
        """
        # get the hidden state of the node and the overall loss
        n_out, loss = self.parent.intercepted_forward_pass(self)

        # first derivative
        dl_df = autograd.grad(loss, n_out, create_graph=True)[0]

        # second derivative for one output (this should algebraically be the same for every output)
        d2l_d2f = autograd.grad(dl_df[0][0], n_out, create_graph=True)[0].view(-1)[0]

        return d2l_d2f

    @property
    def inner_products(self):
        """
        Calculates the L2 inner products of the features over the training set.

        This is done by computing the right term of the hessian decomposition.
        We first build the computation graph by performing a full batch forward pass,
        then accumulating the gradients df(x,θ)/dθ for every x.

        The final matrix is build by calculating the corresponding dot products.

        NOTE:

        The pytorch autograd system can only compute the gradients of scalar leaf tensors.
        This forces us to iterate over every scalar tensor in the output and reconstructing the
        gradients afterwards. This sadly forces us to perform a lot of redundant computations.

        TODO: I see some ways to make this faster. The current time complexity of O(p_size * (outdim * n + p_size))
              can probably be reduced to something like O(p_size * (n + p_size)) due to the symmetry in the hessian

        :return: the l2 inner product matrix
        """
        # perform a forward pass
        out = self.forward_pass()

        n, outdim = out.shape
        p_size = len(self.params)

        grads = []

        # iterate over the output dimension
        for i in range(outdim):
            g = []
            # iterate over the datapoints
            for j in range(n):
                # clear gradients from previous iteration
                self.parent.model.zero_grad()

                # build the autograd graph (this populates the grad field)
                out[j, i].backward(retain_graph=True)

                # df/dθ
                p_grad = torch.tensor([], requires_grad=False)

                # iterate over each parameter (e. g. weight and bias)
                for p in self.parameters():
                    # populate df/dθ
                    p_grad = torch.cat([p_grad, p.grad.view(-1)])

                g.append(p_grad)

            grads.append(torch.stack(g))

        # reconstruct the gradients
        gradients = torch.zeros_like(grads[0])

        for g in grads:
            gradients += g

        # build the l2 inner product matrix (feature orthogonality matrix)
        inner_products = torch.zeros(size=(p_size, p_size))

        for j in range(len(self.params)):
            for k in range(len(self.params)):
                """
                Computes the dot product for 1D tensors. 
                For higher dimensions, sums the product of elements from input 
                and other along their last dimension.
                """
                inner_products[j, k] = torch.inner(gradients[:, j], gradients[:, k])

        return inner_products

    @property
    def hessian(self):
        """
        Calculates the hessian by multiplying the two terms of the decomposition.

        :return: the hessian matrix of the model
        """
        return self.d2l_d2f * self.inner_products

    @property
    def hessian_eigenvalues(self):
        """
        Returns the eigenvalues of the hessian by computing the
        eigenvalue decomposition.

        :return: the eigenvalues and eigenvectors of the model hessian
        """
        return torch.linalg.eigvals(self.hessian).real

    @property
    def hessian_eig_decomposition(self):
        """
        Returns the eigenvalue decomposition of the hessian matrix

        Let A be the hessian. The eigenvalue decomposition is defined as

            A = V * diag(L) * V^-1

        where L are the eigenvalues.

        :return: V, diag(L), V^-1
        """
        # compute the eigenvalues and the eigenvectors of the hessian
        l, v = torch.linalg.eig(self.hessian)

        # the inverse of the basis shift matrix
        v_inverse = torch.linalg.inv(v)

        # the resulting diagonal matrix with the eigenvalues on the diagonal
        diag = torch.diag(l)

        return v.real, diag.real, v_inverse.real

    @property
    def unique_features(self):
        """
        Returns the amount of unique features the model has.
        This is equal to the rank of diag(L).

        NOTE:

        The autograd system will not yield perfect gradients or eigenvalues,
        since it has some numerical instability to it.

        In some cases this can result in having eigenvalues that are not quite zero,
        even though this might algebraically be the case. This is fixed by letting the matrix rank
        computation have a threshold for just viewing entries as zero.

        It is perfectly normal if the calculated rank of the matrix is smaller than it seems when looking at
        the matrix directly.

        :return: The amount of unique features the model has
        """
        _, D, _ = self.hessian_eig_decomposition
        return torch.linalg.matrix_rank(D).item()

    @property
    def orthogonal_basis(self):
        v, d, v_inv = self.hessian_eig_decomposition
        return v_inv

    @property
    def orthogonal_parameters(self):
        return self.orthogonal_basis @ self.params


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
        prev = None

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
        return self.params[0] + self.params[1] * x + self.params[2] * x


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()

        self.dl = DummyLayer()

    def forward(self, x):
        return self.dl(x)


if __name__ == '__main__':
    model = DummyModel()
    # model = SimpleMLP(1, 2, 4, 2, 1)
    inputs = torch.randn(size=(512, 1), dtype=torch.float) * 10
    labels = model(inputs)
    w = MLPWrapper(model, inputs, labels)
    print_modules(w.model)
    print(w[0].unique_features)
    print(w[0].params)
    print(w[0].orthogonal_parameters)
    print(w.orthogonal_model())
