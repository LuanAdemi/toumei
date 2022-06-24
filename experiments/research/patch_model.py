import torch
import torch.nn as nn

from toumei.cnns.objectives.utils import freeze_model
from toumei.models import SimpleMLP


class GradientMask(object):
    def __init__(self, mask):
        self.mask = mask

    def __call__(self, grad, *args, **kwargs):
        return grad.mul(self.mask)


class PatchedModel(nn.Module):
    def __init__(self, zeros=True, weight_lock="model"):
        super(PatchedModel, self).__init__()

        self.patched_m = SimpleMLP(2 * 28 * 28, 28 * 28, 20)

        m_1 = SimpleMLP(28 * 28, (28 * 28) // 2, 10)
        m_1.load_state_dict(torch.load("models/mnist_model.pth"))
        m_2 = SimpleMLP(28 * 28, (28 * 28) // 2, 10)
        m_2.load_state_dict(torch.load("models/mnist_model.pth"))

        self.patch_model(m_1, m_2, zeros=zeros, weight_lock=weight_lock)

        self.linear_layer = nn.Linear(20, 20)

        self.mlp = SimpleMLP(20, 4, 1)
        self.mlp.load_state_dict(torch.load("models/addition_model.pth"))

        freeze_model(self.mlp)

    def forward(self, x):
        x = self.patched_m(x)
        x = self.linear_layer(x)
        x = self.mlp(x)
        return x

    def patch_model(self, m_1, m_2, zeros=True, weight_lock="model", lock_bias=True):

        for named_param1, named_param2, named_parameter in zip(m_1.named_parameters(), m_2.named_parameters(),
                                                               self.patched_m.named_parameters()):
            (key1, param1) = named_param1
            (key2, param2) = named_param2
            (key, parameter) = named_parameter

            if 'weight' in key:
                if zeros:
                    tensor_type = torch.zeros_like
                else:
                    tensor_type = torch.rand_like

                block_1 = tensor_type(param1)
                block_2 = tensor_type(param2)

                half_1 = torch.cat((param1, block_1))
                half_2 = torch.cat((block_2, param2))
                patched_parameter = torch.cat((half_1, half_2), dim=1)

                parameter.data = patched_parameter.data

                if weight_lock == "model":
                    # lock the weights for the whole parameter
                    parameter.requires_grad_(False)
                elif weight_lock == "param":
                    mask_block_1 = torch.cat((torch.zeros_like(param1), torch.ones_like(param1)))
                    mask_block_2 = torch.cat((torch.ones_like(param2), torch.zeros_like(param2)))

                    mask = torch.cat((mask_block_1, mask_block_2), dim=1).cuda()

                    hook = GradientMask(mask)
                    parameter.register_hook(hook)
                elif weight_lock == "none":
                    pass
                else:
                    raise Exception("")

            elif 'bias' in key:
                if lock_bias:
                    parameter.requires_grad_(False)
