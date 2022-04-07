import torch
import torchvision.transforms as T
import torchvision.models as models

# import toumei
import toumei.probe as probe
import toumei.objectives as obj
import toumei.parameterization as param

device = torch.device("cuda")

# the model we want to analyze
alexNet = models.alexnet(pretrained=True)
probe.print_modules(alexNet)

# compose the image transformation for regularization trough transformations robustness
transform = T.Compose([
    T.Pad(12),
    T.RandomRotation((-10, 11)),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # torchvision models need this
])

# define a feature visualization pipeline
fv = obj.Pipeline(
    # the image generator object
    param.Transform(param.FFTImage(1, 3, 256, 256), transform),

    # the objective function
    obj.Channel("features.5:94")
)

# attach the pipeline to the alexNet model
fv.attach(alexNet)

# send the objective to the gpu
fv.to(device)

# optimize the objective
fv.optimize()

# plot the results
fv.generator.plot_image()
