import sys

import toumei.cnns.objectives as obj
import toumei.cnns.parameterization as param
from toumei.models import Inception5h
from toumei.cnns.objectives.utils import set_seed

import torchvision.transforms as T
import torch

import numpy as np

sys.path.append("../")


def test_pixel_image():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(42)

    # compose the image transformation for regularization trough transformations robustness
    transform = T.Compose([
        T.Pad(12),
        T.RandomRotation((-10, 11)),
        T.Lambda(lambda x: x * 255 - 117)  # torchvision models need this
    ])

    model = Inception5h(pretrained=True)

    # define a feature visualization pipeline
    fv = obj.Pipeline(
        # the image generator object
        param.Transform(param.PixelImage(1, 3, 224, 224), transform),

        # the objective function
        obj.Channel("mixed4b_3x3_pre_relu_conv:79")
    )

    # attach the pipeline to the alexNet model
    fv.attach(model)

    # send the objective to the gpu
    fv.to(device)

    # optimize the objective
    fv.optimize(64, tv_loss=True, verbose=False)

    assert (fv.generator.numpy(False) == (np.load("outputs/pixel_image.npy"))).all()
