import torch
import torch.nn as nn


def ntk(model, inp):
    """Calculate the neural tangent kernel of the model on the inputs.
    Returns the gradient feature map along with the tangent kernel.
    """
    out = model(inp)
    p_vec = nn.utils.parameters_to_vector(model.parameters())
    p, = p_vec.shape
    n, outdim = out.shape
    assert outdim == 1, "cant handle output dim higher than 1 for now"

    # this is the transpose jacobian (grad y(w))^T)
    features = torch.zeros(n, p, requires_grad=False)

    for i in range(n):  # for loop over data points
        model.zero_grad()
        out[i].backward(retain_graph=True)
        p_grad = torch.tensor([], requires_grad=False, device=torch.device(inp.device))
        for p in model.parameters():
            p_grad = torch.cat((p_grad, p.grad.reshape(-1)))
        features[i, :] = p_grad

    tk = features @ features.t()  # compute the tangent kernel
    return features, tk