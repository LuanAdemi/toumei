import numpy as np

import torch

from toumei.misc import MLPGraph
from toumei.models import SimpleMLP

device = torch.device("cuda")


def s(n):
    dimensions = [n*2, n]
    x = 1.25
    while n//x > 0:
        dimensions.append(int(n//x))
        x *= 2
    return dimensions

NETWORKS = np.arange(start=232+8, stop=232+27*8, step=8)

network = SimpleMLP(880, 440, 352, 176, 88, 44, 22, 11, 5, 2, 1).to(device)
network.load_state_dict(torch.load("onehot_addition/results_mvg_big/2021_0_MVG.pth"))