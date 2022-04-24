import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import tqdm

import toumei.objectives as obj
import toumei.parameterization as param

from toumei.misc.models.inception5h import Inception5h
from toumei.objectives.misc.utils import freeze_model

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


def mean_l1(a, b):
    return torch.abs(a - b).mean()


def gram_matrix(features, normalize=True):
    C, H, W = features.shape
    features = features.view(C, -1)
    gram = torch.matmul(features, torch.transpose(features, 0, 1))
    if normalize:
        gram = gram / (H * W)
    return gram


class StyleTransferParam(param.ImageGenerator):
    def __init__(self, content_img, style_img):
        super(StyleTransferParam, self).__init__()

        self.shape = content_img.shape[:2]
        self.generator = param.FFTImage(1, 3, *self.shape)

        self.content_input = torch.FloatTensor(np.transpose(content_img, [2, 1, 0]))
        self.style_input = torch.FloatTensor(np.transpose(style_img[:self.shape[0], :self.shape[1], :], [2, 1, 0]))

        self.device = torch.device("cpu")

    def to(self, device: torch.device):
        self.device = device

    @property
    def name(self) -> str:
        return f"StyleTransferParam({self.shape})"

    @property
    def parameters(self) -> list:
        return self.generator.parameters

    def get_image(self, *args, **kwargs) -> torch.Tensor:
        generator_input = self.generator.get_image()[0]
        return torch.stack([generator_input, self.content_input, self.style_input]).to(self.device)


class ActivationDifference(obj.Atom):
    def __init__(self, layers, loss=mean_l1, transform=None):
        super(ActivationDifference, self).__init__("activation_difference", str(layers))

        self.layers = layers
        self.loss = loss

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

    def forward(self, imgs, comp) -> torch.Tensor:

        self.model(imgs)

        base_activations = [atom.activation[comp] for atom in self.activation_atoms]
        if self.transform is not None:
            base_activations = [self.transform(act) for act in base_activations]

        comp_activations = [atom.activation[0] for atom in self.activation_atoms]
        if self.transform is not None:
            comp_activations = [self.transform(act) for act in comp_activations]

        losses = [self.loss(a, b) for a, b in zip(base_activations, comp_activations)]

        return sum(losses)

    @property
    def model(self) -> nn.Module:
        return self.attached_model


class StyleTransfer(obj.Objective):
    def __init__(self,
                 content_image, content_weight,
                 style_image, style_weight,
                 style_layers=None, content_layers=None):
        super(StyleTransfer, self).__init__()

        self.device = torch.device("cpu")

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

        transform = T.Compose([
            T.Pad(12),
            T.RandomRotation((-10, 11)),
            T.Lambda(lambda x: x*255 - 117)
        ])

        self.param = param.Transform(StyleTransferParam(content_image, style_image), transform)
        self.model = Inception5h(pretrained=True)

        self.style_objective.attach(self.model)
        self.content_objective.attach(self.model)

    def to(self, device):
        self.device = device
        self.model.to(self.device)

    @property
    def generator(self) -> param.ImageGenerator:
        return self.param

    def forward(self) -> torch.Tensor:
        imgs = self.param.get_image()
        return self.content_weight * self.content_objective.forward(imgs, 1) \
               + self.style_weight * self.style_objective.forward(imgs, 2)

    def optimize(self, epochs=512, optimizer=torch.optim.Adam, lr=5e-2, tv_loss=False):
        # send the model and the generator to the correct device
        self.model.to(self.device)
        self.model.eval()
        self.generator.to(self.device)

        # attach the optimizer to the parameters of the current generator
        opt = optimizer(self.generator.parameters, lr)

        with tqdm.trange(epochs) as t:
            t.set_description(self.__str__())
            for _ in t:
                def step():
                    # reset gradients
                    opt.zero_grad()

                    # calculate loss using current objective function
                    loss = self.forward()

                    # optimize the generator
                    loss.backward()
                    opt.step()

                    t.set_postfix(loss=loss.item())

                opt.step(step())
