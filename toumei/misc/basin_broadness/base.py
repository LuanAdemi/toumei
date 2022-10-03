import torch
import torch.nn as nn

"""
This script provides the base classes for the feature orthogonalisation algorithm.

This is WIP and will need a lot more work.
"""


class StartNode(nn.Module):
    """
    This dummy object will be the head node of our linked list, which exposes the dataset to the next nodes.
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


class EndNode(nn.Module):
    """
    This dummy object will be the tail node of our linked list.
    """

    def __init__(self, prev: nn.Module = None):
        super(EndNode, self).__init__()
        self.prev = prev


class LinearNode(nn.Module):
    """
    Wraps a linear layer and adds functionality to it
    """
    def __init__(self, name: str, child: nn.Module, prv: nn.Module = None, nxt: nn.Module = None):
        super(LinearNode, self).__init__()

        # the node name (the name of the wrapped module)
        self.name = name

        # the wrapped linear layer
        self.module = child

        # the previous node
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
        """
        Returns the weight matrix of the current node

        :return: A matrix containing the weights of the node
        """
        return next(self.parameters())

    @property
    def biases(self):
        """
        Returns the bias vector of the current node
        (Please ignore the code below. It will be better for your sanity)

        :return: A vector containing the bias
        """
        for p in self.parameters():
            continue
        return p

    @property
    def params(self):
        """
        A matrix containing the parameters of the layer.

        This is essentially the weight matrix with one additional column and row like so

        W_11 . . . W_1j B_1
         .   .           .
         .      .        .
         .         .     .
        W_i1 . . . W_ij B_i
        0    . . .   0   1 < One-Row
                         ^
                        Bias

        This creates a new neuron for the bias, which can be used as a vector (function) for the hilbert space

        :return: the parameter matrix
        """
        weights = self.weights
        biases = self.biases.unsqueeze(1)

        params = torch.cat([weights, biases], dim=1)
        one_row = torch.tensor([0 for _ in range(weights.shape[1])] + [1]).unsqueeze(0)

        params = torch.cat([params, one_row], dim=0)
        return params

    @property
    def activations(self):
        """
        Returns the activations of the current node

        :return: the activation vector
        """
        return self.prev.forward_pass()

    @property
    def activation_matrix(self):
        """
        Computes the activation matrix by calculating the outer product of the activation vector with its transpose

        :return: The activation matrix
        """
        # retrieve the activations
        act = self.activations

        # append ones to the activation vector, to represent the bias as a constant feature in the hilbert space
        bias_row = torch.ones(size=(act.shape[0], 1))
        act = torch.cat([act, bias_row], dim=1)
        matrix = torch.zeros(size=(act.shape[1], act.shape[1]))

        # sum over all dataset samples
        for x in act:
            matrix += torch.outer(x, x.T)

        return matrix

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
        L, Q = torch.linalg.eig(self.activation_matrix)

        # the resulting diagonal matrix with the eigenvalues on the diagonal

        return Q.real, L.real, torch.linalg.inv(Q).real

    @property
    def orthogonal_basis(self):
        """
        Returns the orthogonal (eigen) basis of the parameter space

        :return: the orthogonal basis
        """
        Q, D, Q_inv = self.act_eig_decomposition
        return Q, D, Q_inv

    def orthogonalise(self):
        """
        Orthonormalizes the weight matrix of the current node.

        This is done by computing the eigen-decomposition of the activation matrix in order to find a basis in which
        said matrix has a diagonal form.

        The weight matrix is transformed as follows:

        W' = D @ W_i @ Q_i^-1 @ Q_i

        :return: The orthogonal parameters for this node
        """
        # collect the orthogonal basis of the current node
        Q, L, Q_INV = self.orthogonal_basis

        # note: 1-d tensors are for some reason ALWAYS row vectors, so this transpose hack is needed for rescaling
        ortho_weights = (torch.diag(L) @ (self.params @ Q_INV @ Q).T).T

        ortho_module = nn.Linear(*ortho_weights.T.shape, bias=False)

        ortho_module.weight = nn.Parameter(ortho_weights)

        print(ortho_weights)

        return ortho_module, L


class MLPWrapper(nn.Module):
    """
    Implements a linked array list like structure.

    Nodes (LinearLayers) are stored in a double linked array list and can be accessed dynamically.

    A node has access to every previous and next node, so it can dynamically build the computation graph.
    """
    def __init__(self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor, loss_func=nn.MSELoss()):
        super(MLPWrapper, self).__init__()

        # the dataset and loss function
        self.X = inputs
        self.Y = labels
        self.loss_func = loss_func

        # the model
        self.model = model

        # a container for the single linked array list
        self.nodes = []

        # vectorized model parameters
        self.params = nn.utils.parameters_to_vector(self.model.parameters())

        # first dummy object
        prev = StartNode(self.X)

        # fill the linked list
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # create new node
                node = LinearNode(name, module, prev)
                # set the next node of the previous node to the new node
                prev.next = node
                # add the new node to the node list
                self.nodes.append(node)
                # set the previous node to the new node
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
        # call a forward pass on the tail node
        return self.nodes[-1].forward_pass()

    def forward_pass(self):
        """
        Performs a forward pass

        :return: the model output and the loss
        """
        out = self.forward()
        loss = self.loss_func(out, self.Y)
        return out, loss

    def orthogonal_model(self):
        """
        This is the main algorithm.
        It collects the orthogonal features of each node (layer) and builds the corresponding orthogonal model.

        :param inplace  if true, the algorithm will be performed on the passed model, else it will work on a copy
        :return: the orthogonal model
        """

        modules = []

        current_node = 0

        # iterate over each module of the orthogonal model
        for name, module in self.model.named_children():
            if isinstance(module, nn.Linear):
                # set the parameters to the orthogonal ones
                ortho_module, L = self[current_node].orthogonalise()
                modules.append(ortho_module)
                current_node += 1
                continue

            modules.append(module)

        return nn.Sequential(*modules)

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
