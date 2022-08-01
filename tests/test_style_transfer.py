import numpy as np
import torch
from PIL import Image
import sys

from toumei.misc import StyleTransfer
from toumei.cnns.featurevis.objectives.utils import set_seed

sys.path.append("../")

def test_style_transfer():
    content = np.asarray(Image.open("../assets/big_ben.png")) / 255
    style = np.asarray(Image.open("../assets/style.png")) / 255

    set_seed(42)

    objective = StyleTransfer(content, 250, style, 1)

    objective.to(torch.device("cuda"))

    objective.optimize(64, verbose=False)

    assert (objective.generator.numpy(False) == (np.load("outputs/style_transfer.npy"))).all()
