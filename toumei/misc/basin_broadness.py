from collections.abc import Iterator

import torch
import torch.nn as nn
from torch import autograd
from torch.nn import Parameter

from toumei.models import SimpleMLP


class LinearNode(nn.Module):
    """
    Wraps a linear layer and adds functionality to it
    """
    def __init__(self, parent: nn.Module, child: nn.Module, prv: nn.Module = None):
        super(LinearNode, self).__init__()

        self.parent = parent
        self.child = child

        self.prev = prv

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return self.child.parameters()

    def forward(self, x):
        return self.child.forward(x)

    def forward_pass(self):
        if self.prev is not None:
            return self.forward(self.prev.forward_pass())
        else:
            return self.forward(self.parent.X)

    @property
    def weights(self):
        return next(self.parameters())

    @property
    def d2l_d2f(self):
        """
        Calculates the first term of the hessian decomposition.

        :return: the corresponding gradient
        """
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

        :return: the l2 inner product matrix
        """
        # perform a forward pass
        out = self.forward_pass()

        n, outdim = out.shape
        p_size = len(self.weights)

        grads = []

        for o in out.T:
            g = []
            for i in range(n):
                # clear gradients from previous iteration
                self.parent.model.zero_grad()

                # build the autograd graph
                o[i].backward(retain_graph=True)

                # df(x,θ)/dθ
                g.append(self.weights.grad.view(-1))

            grads.append(torch.stack(g))

        gradients = torch.zeros_like(grads[0])

        # cumulate the gradients
        for g in grads:
            gradients += g

        inner_products = torch.zeros(size=(p_size, p_size))

        # build the l2 inner product matrix (feature orthogonality matrix)
        for j in range(len(self.weights)):
            for k in range(len(self.weights)):
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
        return torch.linalg.eigvals(self.hessian)

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

        return v, diag, v_inverse

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


class MLPWrapper(nn.Module):
    def __init__(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor, loss_func=nn.MSELoss()):
        super(MLPWrapper, self).__init__()

        self.X = x
        self.Y = y
        self.loss_func = loss_func

        self.model = model

        self.nodes = []

        prev = None
        for key, value in self.model.named_modules():
            if type(value) == nn.Linear:
                print(key)
                node = LinearNode(self, value, prev)
                self.nodes.append(node)
                prev = node

    def forward(self):
        return self.nodes[-1].forward_pass()

    def forward_pass(self):
        out = self.forward()
        loss = self.loss_func(out, self.Y)
        return out, loss

    def intercepted_forward_pass(self, node):
        n = self.nodes.index(node)
        n_out = self.nodes[n].forward_pass()

        out = n_out
        for node in self.nodes[n+1:]:
            out = node(out)

        loss = self.loss_func(out, self.Y)
        return n_out, loss


if __name__ == '__main__':
    model = SimpleMLP(8, 16, 8, 4, 2, 1)
    inputs = torch.randn(size=(512, 8), dtype=torch.float)
    labels = model(inputs)
    w = MLPWrapper(model, inputs, labels)
    v, d, t = w.nodes[1].hessian_eig_decomposition
    print(w.nodes[1].inner_products)
    print(w.nodes[1].unique_features)
