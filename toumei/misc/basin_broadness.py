import torch
import torch.nn as nn
from torch import autograd


class BasinVolumeMeasurer(object):
    """
    A class implementing some ideas of the document on basin broadness.

    WARNING: Advanced math and autograd usage incoming.
    """
    def __init__(self, model, inputs, labels, loss_func=nn.MSELoss()):
        self.model = model
        self.parameters = nn.utils.parameters_to_vector(model.parameters())
        self.inputs = inputs
        self.labels = labels
        self.loss_func = loss_func

    def forward_pass(self):
        """
        Performs a forward pass

        :return: the model output
        """
        return self.model(self.inputs)

    def get_loss(self):
        """
        Performs a forward pass and calculates the loss

        :return: the model output and the loss
        """
        out = self.forward_pass()
        return out, self.loss_func(out, self.labels)

    def calculate_d2l_d2f(self):
        """
        Calculates the first term of the hessian decomposition

        :return: the corresponding gradient
        """
        output, loss = self.get_loss()

        # first derivative
        dl_df = autograd.grad(loss, output, create_graph=True)[0]

        d2l_d2f = []

        # iterate over every gradient of the first derivative
        for i, (grad, out) in enumerate(zip(dl_df, output)):
            # second derivative
            drv = autograd.grad(grad, output, create_graph=True)[0].view(-1)
            d2l_d2f.append(drv[i])

        # return the first element, since every entry is the same
        return d2l_d2f[0]

    def calculate_inner_product_matrix(self):
        """
        Calculates the L2 inner products of the features over the training set.

        This is done by computing the right term of the hessian decomposition.
        We first build the computation graph by performing a full batch forward pass,
        then accumulating the gradients df(x,θ)/dθ for every x.

        The final matrix is build by calculating the corresponding dot products.

        :return: the l2 inner product matrix
        """
        out = self.forward_pass()
        n, outdim = out.shape
        p_size = len(self.parameters)

        grads = []

        for i in range(n):
            # clear gradients from previous iteration
            self.model.zero_grad()

            # build the autograd graph
            out[i].backward(retain_graph=True)
            p_grad = torch.tensor([], requires_grad=False)

            # iterate over every model parameter
            for p in self.model.parameters():
                grad = p.grad # df(x,θ)/dθ
                p_grad = torch.cat((p_grad, grad.reshape(-1)))  # append the gradient
            grads.append(p_grad)

        grads = torch.stack(grads)
        inner_products = torch.zeros(size=(p_size, p_size))

        # build the l2 inner product matrix (feature orthogonality matrix)
        for j in range(len(self.parameters)):
            for k in range(len(self.parameters)):
                """
                Computes the dot product for 1D tensors. 
                For higher dimensions, sums the product of elements from input 
                and other along their last dimension.
                """
                inner_products[j, k] = torch.inner(grads[:, j], grads[:, k])

        return inner_products

    def calculate_hessian(self):
        """
        Calculates the hessian by multiplying the two terms of the decomposition.

        :return: the hessian matrix of the model
        """
        return self.calculate_d2l_d2f() * self.calculate_inner_product_matrix()

    def get_hessian_eigvals(self):
        """
        Returns the eigenvalues of the hessian by computing the
        eigenvalue decomposition.

        :return: the eigenvalues and eigenvectors of the model hessian
        """
        return torch.linalg.eigvals(self.calculate_hessian())

    def get_hessian_eig_decomposition(self):
        """
        Returns the eigenvalue decomposition of the hessian matrix

        Let A be the hessian. The eigenvalue decomposition is defined as

            A = V * diag(L) * V^-1

        where L are the eigenvalues.

        :return: V, diag(L), V^-1
        """
        l, v = torch.linalg.eig(self.calculate_hessian())
        # the inverse of the eigenvector space
        v_inverse = torch.linalg.inv(v)
        diag = torch.diag(l)

        return v, diag, v_inverse

    def unique_features(self):
        """
        Returns the amount of unique features the model has.
        This is equal to the matrix rank of diag(L).

        :return: The amount of unique features the model has
        """
        _, D, _ = self.get_hessian_eig_decomposition()
        return torch.linalg.matrix_rank(D).item()


class DummyModel(nn.Module):
    """
    The little example presented in the document
    """
    def __init__(self):
        super(DummyModel, self).__init__()

        # these are the parameters for optimization
        self.param = nn.Parameter(torch.tensor([1, 1, 1], dtype=torch.float))

    def forward(self, z):
        return self.param[0] + self.param[1] * z + self.param[2] * torch.cos(z)


if __name__ == '__main__':
    model = DummyModel()
    inputs = torch.tensor([[0.], [1.], [2.], [3.], [4.], [5.], [6.]], dtype=torch.float)
    labels = model(inputs)
    measurer = BasinVolumeMeasurer(model, inputs, labels)
    V, D, V_inv = measurer.get_hessian_eig_decomposition()
    print(D)
    print(measurer.unique_features())

