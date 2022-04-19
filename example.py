import torch
import torchvision.transforms as T
from lucent.modelzoo import inceptionv1

# import toumei
import toumei.probe as probe
import toumei.objectives as obj
import toumei.parameterization as param

device = torch.device("cuda")

# the model we want to analyze
model = inceptionv1(pretrained=True)
probe.print_modules(model)

# define a feature visualization pipeline
fv = obj.Pipeline(
    # the image generator object
    param.Neural(1, 3, 256, 256),

    # the objective function
    obj.Channel("mixed4d_3x3_bottleneck_pre_relu_conv:139")
)

# attach the pipeline to the alexNet model
fv.attach(model)

# send the objective to the gpu
fv.to(device)

# optimize the objective
fv.optimize()

# plot the results
fv.generator.plot_image()

