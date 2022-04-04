import torch.nn

import toumei.probe as probe
from toumei.objectives import Pipeline
import toumei.objectives.atoms as obj
import toumei.parameterization as param
import torchvision.models as models
import torchvision.transforms as T

device = torch.device("cuda")


# the model we want to analyze
alexNet = models.alexnet(pretrained=True)
probe.print_modules(alexNet)

# define a feature visualization pipeline
fv = Pipeline(
    # the image generator object
    param.Transform(param.PixelImage(1, 3, 512, 512)),

    # the objective function
    obj.Channel("features.8:12")
)
# attach the pipeline to the alexNet model
fv.attach(alexNet)

# send the objective to the gpu
fv.to(device)

# optimize the objective
fv.optimize()

# plot the results
fv.generator.plot_image()
