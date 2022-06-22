import torch
import torch.nn as nn

from toumei.cnns.objectives.utils import freeze_model
from toumei.models import SimpleMLP


class PatchedModel(nn.Module):
    def __init__(self):
        super(PatchedModel, self).__init__()

        self.patched_m = SimpleMLP(2 * 28 * 28, 28 * 28, 20)

        m_1 = SimpleMLP(28 * 28, (28 * 28) // 2, 10)
        m_1.load_state_dict(torch.load("mnist_model.pth"))
        m_2 = SimpleMLP(28 * 28, (28 * 28) // 2, 10)
        m_2.load_state_dict(torch.load("mnist_model.pth"))

        self.patch_model(m_1, m_2)

        self.linear_layer = nn.Linear(20, 20)

        self.mlp = SimpleMLP(20, 4, 1)
        self.mlp.load_state_dict(torch.load("addition_model.pth"))

        freeze_model(self.mlp)

    def forward(self, x):
        x = self.patched_m(x)
        x = self.linear_layer(x)
        x = self.mlp(x)
        return x

    def patch_model(self, m_1, m_2, zeros=True, lock_weights=True):
        for named_param1, named_param2, named_parameter in zip(m_1.named_parameters(), m_2.named_parameters(),
                                                               self.patched_m.named_parameters()):
            (key1, param1) = named_param1
            (key2, param2) = named_param2
            (key, parameter) = named_parameter
            if lock_weights:
                # lock the weights (gradients will automatically be set to 0)
                parameter.requires_grad_(False)
            if 'weight' in key:
                if zeros:
                    block_1 = torch.zeros_like(param1)
                    block_2 = torch.zeros_like(param2)
                else:
                    block_1 = torch.rand_like(param1)
                    block_2 = torch.rand_like(param2)

                patched_parameter = torch.cat((torch.cat((param1, block_1)), torch.cat((param2, block_2))), dim=1)
                parameter.data = patched_parameter.data
            elif 'bias' in key:
                continue
