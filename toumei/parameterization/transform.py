from toumei.parameterization import Generator
import torch.nn as nn
import torch
import torchvision.transforms as T

standard_transform = torch.nn.Sequential(
    T.Pad(12),
    T.ColorJitter(8),
    T.RandomRotation((-10, 11)),
    T.ColorJitter(4)
)


class Transform(Generator):
    def __init__(self, generator:Generator, transform_func: nn.Module = standard_transform):
        super(Transform, self).__init__()
        self.img_generator = generator
        self.transform_function = transform_func

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
