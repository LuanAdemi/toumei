from toumei.misc.styletransfer import StyleTransfer
from PIL import Image
import torch

import numpy as np

content = np.asarray(Image.open("assets/gw.jpg")) / 255
style = np.asarray(Image.open("assets/2997.jpg")) / 255

sf = StyleTransfer(content, 200, style, 1)

sf.to(torch.device("cuda"))

sf.optimize()

sf.generator.plot_image()
