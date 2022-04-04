import torch

import toumei.probe as probe
from toumei.objectives import Pipeline
import toumei.objectives.atoms as obj
import toumei.parameterization as param
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np


def tensor_to_img_array(tensor):
    image = tensor.cpu().detach().numpy()
    image = np.transpose(image, [0, 2, 3, 1])
    return image[0]


alexNet = models.alexnet(pretrained=True)
probe.print_modules(alexNet)

# define a feature visualization pipeline
fv = Pipeline(
    # the image generator object
    param.PixelImage(1, 3, 512, 512),

    # the objective function
    obj.Channel("features.0:0")
)
# attach the pipeline to the alexNet model
fv.attach(alexNet)
fv.optimize(500)
plt.imshow(tensor_to_img_array(fv.generator.getImage()))
plt.show()
