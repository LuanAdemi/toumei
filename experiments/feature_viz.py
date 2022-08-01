import torch
import torchvision.transforms as T
from toumei.models import Inception5h

# import toumei
import toumei.cnns.objectives as obj
import toumei.cnns.featurevis.parameterization as param

from toumei.cnns.objectives.utils import set_seed

"""
Performing feature visualisation on the Inception model 
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set the seed for reproducibility
set_seed(42)

# compose the image transformation for regularization through transformations robustness
transform = T.Compose([
    T.Pad(12),
    T.RandomRotation((-10, 11)),
    T.Lambda(lambda x: x*255 - 117)  # inception needs this
])

# the model we want to analyze
model = Inception5h(pretrained=True)

# define a feature visualization pipeline
fv = obj.Pipeline(
    # the image generator object
    param.Transform(param.FFTImage(1, 3, 224, 224), transform),

    # the objective function
    obj.Channel("mixed3a:74")
)

# attach the pipeline to the alexNet model
fv.attach(model)

# send the objective to the gpu
fv.to(device)

# optimize the objective
fv.optimize()

# plot the results
fv.plot()
