from copy import deepcopy

import torch.nn as nn
import torch


class StartNode(nn.Module):
    """
    This will be the head node of our linked list, which exposes the dataset to the next nodes
    """
    def __init__(self, data, nxt: nn.Module = None):
        super(StartNode, self).__init__()

        self.data = data
        self.next = nxt

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
        # this is the head node, so no preceding nodes exist
        return self.data

    @property
    def orthogonal_basis(self):
        """
        Returns the orthogonal (eigen) basis of the parameter space

        :return: the orthogonal basis
        """
        v = torch.eye(self.next.weights.shape[1])
        return v, v


class EndNode(nn.Module):
    """
    This will be the tail node of our linked list, which exposes the dataset to the next nodes
    """

    def __init__(self, prev: nn.Module = None):
        super(EndNode, self).__init__()
        self.prev = prev


class LinearNode(nn.Module):
    """
    Wraps a linear layer and adds functionality to it
    """
    def __init__(self, name: str, parent: nn.Module, child: nn.Module, prv: nn.Module = None, nxt: nn.Module = None):
        super(LinearNode, self).__init__()

        # the node name (the name of the wrapped module)
        self.name = name

        # the parent container
        self.parent = parent

        # the wrapped linear layer
        self.module = child

        # the preceding node
        self.prev = prv

        # the next node
        self.next = nxt

    def parameters(self, recurse: bool = True):
        return self.module.parameters()

    def forward(self, x):
        """
        Forwards a tensor through the node

        :param x: the input tensor
        :return: the hidden state/output
        """
        return self.module.forward(x)

    def forward_pass(self):
        """
        Performs a forward pass by making recursive calls to all previous nodes

        :return: the hidden state of this node
        """
        return self.forward(self.prev.forward_pass())

    @property
    def weights(self):
        return next(self.parameters())

    @property
    def biases(self):
        for p in self.parameters():
            ""
        return p

    @biases.setter
    def biases(self, value):
        self._biases = value

    @property
    def params(self):
        return torch.cat([self.weights, self.biases.unsqueeze(1)], dim=1)

    @property
    def activations(self):
        return self.prev.forward_pass()

    @property
    def activation_matrix(self):
        # retrieve the activations
        act = self.activations

        # append a one to the activation vector, to represent the bias as a constant feature in the hilbert space
        act = torch.cat([act, torch.ones((act.shape[0], 1))], dim=1)
        matrix = torch.zeros(size=(act.shape[1], act.shape[1]))

        # sum over all dataset samples
        for a in act:
            matrix += torch.outer(a, a.T)

        return matrix

    @property
    def act_eigenvalues(self):
        """
        Returns the eigenvalues of the hessian by computing the
        eigenvalue decomposition.

        :return: the eigenvalues and eigenvectors of the model hessian
        """
        return torch.linalg.eigvals(self.activation_matrix).real

    @property
    def act_eig_decomposition(self):
        """
        Returns the eigenvalue decomposition of the hessian matrix

        Let A be the hessian. The eigenvalue decomposition is defined as

            A = V * diag(L) * V^-1

        where L are the eigenvalues.

        Note:

        The eigenvectors are normalized to have norm 1 by default (see pytorch documentation).

        :return: V, diag(L), V^-1
        """
        # compute the eigenvalues and the eigenvectors of the hessian
        l, v = torch.linalg.eig(self.activation_matrix)

        # the (pseudo) inverse of the basis shift matrix
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
        _, D, _ = self.act_eig_decomposition
        return torch.linalg.matrix_rank(D).item()

    @property
    def orthogonal_basis(self):
        """
        Returns the orthogonal (eigen) basis of the parameter space

        :return: the orthogonal basis
        """
        v, d, v_inv = self.act_eig_decomposition
        return v, d, v_inv

    def orthogonalise(self):
        v, d, v_inv = self.orthogonal_basis
        v_n, d_n, v_inv_n = self.next.orthogonal_basis

        print(v.shape, v_n.shape)

        # set the biases to zero
        ortho_bias = torch.zeros_like(self.biases)

        weights = torch.cat([self.weights, self.biases.unsqueeze(1)], dim=1)

        ortho_weights = d @ v_n @ weights @ v_inv

        print(ortho_weights)


class MLPWrapper(nn.Module):
    """
    Implements a linked array list like structure.

    Nodes (LinearLayers) are stored in a double linked array list and can be accessed dynamically.

    A node has access to every preceding node, so it can dynamically build the computation graph.
    """
    def __init__(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor, loss_func=nn.MSELoss()):
        super(MLPWrapper, self).__init__()

        # the dataset and loss function
        self.X = x
        self.Y = y
        self.loss_func = loss_func

        # the model
        self.model = deepcopy(model)
        self.old_model = model

        # a container for the single linked array list
        self.nodes = []

        # vectorized model parameters
        self.params = nn.utils.parameters_to_vector(self.model.parameters())

        # first dummy object
        prev = StartNode(self.X)

        # fill the linked list
        for key, value in self.model.named_modules():
            if isinstance(value, nn.Linear):
                node = LinearNode(key, self, value, prev, nxt=None)
                prev.next = node
                self.nodes.append(node)
                prev = node

        # last dummy object
        prev.next = EndNode(prev=prev)

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
            ortho_model = self.model
        else:
            ortho_model = deepcopy(self.model)

        current_node = 0

        # iterate over each module of the orthogonal model
        for name, module in ortho_model.named_modules():
            if isinstance(module, nn.Linear):
                # set the parameters to the orthogonal ones
                ortho_param = self[current_node].orthogonal_parameters

                weights = next(module.parameters())
                weights.data = ortho_param

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