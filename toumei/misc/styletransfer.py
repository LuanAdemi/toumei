import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import tqdm

import toumei
import toumei.cnns.featurevis.objectives as obj
import toumei.cnns.featurevis.parameterization as param

from toumei.cnns.featurevis.feature_visualization_method import FeatureVisualizationMethod

from toumei.models import Inception5h
from toumei.cnns.featurevis.objectives.utils import freeze_model

from toumei.cnns.featurevis.parameterization.imagegenerator import ImageGenerator

# standard layers for the style transfer
standard_style_layers = [
    'conv2d2',
    'mixed3a',
    'mixed4a',
    'mixed4b',
    'mixed4c',
]

standard_content_layers = [
    'mixed3b',
]


def gram_matrix(features, normalize=True):
    """
    Calculates the Gram matrix for the given features

    :param features: the input features
    :param normalize: normalize the gram matrix
    :return:
    """
    C, H, W = features.shape
    features = features.view(C, -1)
    gram = torch.matmul(features, torch.transpose(features, 0, 1))
    if normalize:
        gram = gram / (H * W)
    return gram


class StyleTransferParam(ImageGenerator):
    """
    A ImageGenerator for style transfer.
    It uses a FFTImage generator under the hood and exposes its parameters.
    """
    def __init__(self, content_img, style_img):
        """
        Initializes a new StyleTransferParam generator

        :param content_img: the content image
        :param style_img: the style image
        """
        super(StyleTransferParam, self).__init__()

        self.shape = content_img.shape[:2]
        self.generator = param.FFTImage(1, 3, *self.shape, saturation=20.0)

        self.content_input = torch.FloatTensor(np.transpose(content_img, [2, 0, 1]))
        self.style_input = torch.FloatTensor(np.transpose(style_img[:self.shape[0], :self.shape[1], :], [2, 0, 1]))

        self.device = torch.device("cpu")

    def to(self, device: torch.device):
        self.device = device

    @property
    def name(self) -> str:
        return f"StyleTransferParam({self.shape})"

    @property
    def parameters(self) -> list:
        """
        Expose the FFTImage generators parameters

        :return:
        """
        return self.generator.parameters

    def get_image(self, *args, **kwargs) -> torch.Tensor:
        """
        Returns a tensor which contains the generator, content and style image

        :param args: the arguments
        :param kwargs: the keyword arguments
        :return: the stacked images
        """
        generator_input = self.generator.get_image()[0]
        return torch.stack([generator_input, self.content_input, self.style_input]).to(self.device)


class ActivationDifference(obj.Atom):
    """
    A custom objective atom for style transfer.
    It computes the activation difference between the content and style images using the given layers and metric.
    """
    def __init__(self, layers, loss=nn.L1Loss(), transform=None):
        """
        Initializes a new ActivationDifference atom

        :param layers: the layers for the difference computation
        :param loss: the loss metric function
        :param transform: a transform function, default is None
        """
        super(ActivationDifference, self).__init__("activation_difference", str(layers))

        self.layers = layers
        self.loss = loss

        # create layer objectives for the specified layers
        self.activation_atoms = [obj.Layer(layer) for layer in self.layers]
        self.hooks = []

        self.attached_model = None
        self.transform = transform

    def attach(self, model: nn.Module):
        freeze_model(model)
        for atom in self.activation_atoms:
            self.hooks.append(atom.attach(model))

        self.attached_model = model

    def detach(self):
        for hook in self.hooks:
            if hook is not None:
                self.hook.remove()

    def forward(self, images, comp) -> torch.Tensor:
        """
        Computed the activation difference between the transfer image and the given comparison image.
        Also performs transforms if a transform function was specified.

        :param images: the images generated using the StyleTransferParam generator
        :param comp: the index of the comparison image
        :return: the loss using the specified metric
        """

        # pass the images through the model
        self.model(images)

        # the activation of the comparison image
        comp_activations = [atom.activation[comp] for atom in self.activation_atoms]
        if self.transform is not None:
            comp_activations = [self.transform(act) for act in comp_activations]

        # the activation of the transfer image
        transfer_activations = [atom.activation[0] for atom in self.activation_atoms]
        if self.transform is not None:
            transfer_activations = [self.transform(act) for act in transfer_activations]

        # calculate the loss using the specified metric
        losses = [self.loss(a, b) for a, b in zip(comp_activations, transfer_activations)]

        return sum(losses)

    @property
    def model(self) -> nn.Module:
        return self.attached_model


class StyleTransfer(FeatureVisualizationMethod):
    """
    A custom objective for style transfer using InceptionV1
    """
    def __init__(self,
                 content_image, content_weight,
                 style_image, style_weight,
                 content_layers=None, style_layers=None):
        """
        Initializes a new StyleTransfer objective

        :param content_image: the content image
        :param content_weight: the content weight
        :param style_image: the style image
        :param style_weight: the style weight
        :param content_layers: the layers used for the content activation
        :param style_layers: the layers used for the style activation
        """
        super(StyleTransfer, self).__init__()

        self.device = torch.device("cpu")

        # set the layers to default values if None
        if content_layers is None:
            content_layers = standard_content_layers
        if style_layers is None:
            style_layers = standard_style_layers

        self.content_img = content_image
        self.content_weight = content_weight
        self.content_layers = content_layers
        self.content_objective = ActivationDifference(self.content_layers)

        self.style_img = style_image
        self.style_weight = style_weight
        self.style_layers = style_layers
        self.style_objective = ActivationDifference(self.style_layers, transform=gram_matrix)

        # standard transforms
        transform = T.Compose([
            T.Pad(12),
            T.RandomRotation((-10, 11)),
            T.Lambda(lambda x: x*255 - 117)
        ])

        # generator and inception model for the style transfer
        self.param = param.Transform(StyleTransferParam(content_image, style_image), transform)
        self.model = Inception5h(pretrained=True)

        # attach the atoms to the inception model
        self.style_objective.attach(self.model)
        self.content_objective.attach(self.model)

    def to(self, device):
        self.device = device
        self.model.to(self.device)

    @property
    def generator(self) -> ImageGenerator:
        return self.param

    def forward(self) -> torch.Tensor:
        # get the current images from the generator
        images = self.param.get_image()

        # the loss function
        loss = self.content_weight * self.content_objective.forward(images, 1) \
               + self.style_weight * self.style_objective.forward(images, 2)

        return loss

    def optimize(self, epochs=512, optimizer=torch.optim.Adam, lr=5e-2, tv_loss=False, verbose=True):
        # send the model and the generator to the correct device
        self.model.to(self.device)
        self.model.eval()
        self.generator.to(self.device)

        # attach the optimizer to the parameters of the current generator
        opt = optimizer(self.generator.parameters, lr)

        with tqdm.trange(epochs, disable=not verbose) as t:
            t.set_description(self.__str__())
            for _ in t:
                def step():
                    # reset gradients
                    opt.zero_grad()

                    # calculate loss using current objective function
                    loss = self.forward()  # (forward pass is performed in self.forward())

                    # optimize the generator
                    loss.backward()
                    opt.step()

                    t.set_postfix(loss=loss.item())

                opt.step(step())
