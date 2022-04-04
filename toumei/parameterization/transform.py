from toumei.parameterization import Generator
import torch.nn as nn
import torch


class Transform(Generator):
    def __init__(self, generator:Generator, transform: nn.Module):
        super(Transform, self).__init__()
        self.img_generator = generator
        self.transform_function = transform

    @property
    def name(self) -> str:
        return f"Transform()"

    @property
    def parameters(self) -> torch.Tensor:
        return self.img_generator.parameters

    def get_image(self, transform=True) -> torch.Tensor:
        if transform:
            return self.transform_function(self.img_generator.get_image())
        else:
            return self.img_generator.get_image()
