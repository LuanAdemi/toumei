import torch
import torchvision.transforms as T
from toumei.models import Inception5h

# import toumei
import toumei.objectives as obj
import toumei.parameterization as param

from toumei.objectives.utils import set_seed

device = torch.device("cuda")

# set the seed for reproducibility
set_seed(42)

# compose the image transformation for regularization trough transformations robustness
transform = T.Compose([
    T.Pad(12),
    T.RandomRotation((-10, 11)),
    T.Lambda(lambda x: x*255 - 117)  # torchvision models need this
])

# the model we want to analyze
model = Inception5h(pretrained=True)

# define a feature visualization pipeline
fv = obj.Pipeline(
    # the image generator object
    param.Transform(param.FFTImage(1, 3, 224, 224), transform),

    # the objective function
    obj.Channel("mixed4b_3x3_pre_relu_conv:79")
)

# attach the pipeline to the alexNet model
fv.attach(model)

# send the objective to the gpu
fv.to(device)

# optimize the objective
fv.optimize()

# plot the results
fv.generator.plot_image()

