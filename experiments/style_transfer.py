from toumei.misc.styletransfer import StyleTransfer
from toumei.cnns.objectives.utils import set_seed
from PIL import Image
import torch

import numpy as np

content = np.asarray(Image.open("../assets/big_ben.png")) / 255
style = np.asarray(Image.open("../assets/style.png")) / 255

set_seed(42)

sf = StyleTransfer(content, 250, style, 1)

sf.to(torch.device("cuda"))

sf.optimize()

sf.generator.plot_image()
