import torch

import toumei.probe as probe
from toumei.objectives import Pipeline
import toumei.objectives.atoms as obj
import toumei.parameterization as param
import torchvision.models as models

alexNet = models.alexnet()
probe.print_modules(alexNet)

# define a feature visualization pipeline
fv = Pipeline(
    # the image generator object
    param.Generator(),

    # the objective function
    obj.Channel("features.0:0")
)
# attach the pipeline to the alexNet model
fv.attach(alexNet)
print(fv)
# forward-pass
x = torch.rand((1, 3, 512, 512), requires_grad=True)
print(fv(x))
