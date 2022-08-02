import torch
import torch.nn as nn


class BasinVolumeMeasurer(object):
    def __init__(self, model, loss_fc, data):
        self.model = model
        self.parameters = nn.utils.parameters_to_vector(model.parameters())
        self.loss_fc = loss_fc
        self.data = data

    def forward_pass(self):
        return self.model(self.data)

    def calculate_inner_products(self):
        out = self.forward_pass()
        n, outdim = out.shape
        p_size = len(self.parameters)

        grads = []

        for i in range(n):
            self.model.zero_grad()
            out[i].backward(retain_graph=True)
            p_grad = torch.tensor([], requires_grad=False)
            for p in self.model.parameters():
                # df(x,θ)/dθ
                p_grad = torch.cat((p_grad, p.grad.reshape(-1)))
            grads.append(p_grad)

        grads = torch.stack(grads)
        inner_products = torch.zeros(size=(p_size, p_size))

        for j in range(len(self.parameters)):
            for k in range(len(self.parameters)):
                inner_products[j, k] = torch.inner(grads[:, j], grads[:, k])

        return inner_products


class DummyModel(nn.Module):
    def __init__(self, theta):
        super(DummyModel, self).__init__()

        # these are the parameters for optimization
        self.param = nn.Parameter(torch.tensor(theta, dtype=torch.float))

    def forward(self, z):
        return self.param[0] + self.param[1] * z + self.param[2] * torch.cos(z)


dummy = DummyModel([1, 1, 1])
measurer = BasinVolumeMeasurer(dummy, None, torch.tensor([[0], [2], [3]], dtype=torch.float))
print(measurer.calculate_inner_products())
print(torch.dot(torch.tensor([1, 0]), torch.tensor([0, 1])))
