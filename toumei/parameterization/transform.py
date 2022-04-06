from toumei.parameterization import Generator
import torch.nn as nn
import torch
import torchvision.transforms as T

standard_transform = T.Compose([
    T.Pad(12),
    T.RandomRotation((-10, 11))
])


class Transform(Generator):
    def __init__(self, generator: Generator, transform_func=standard_transform):
        super(Transform, self).__init__()
        self.img_generator = generator
        self.transform_function = transform_func

    @property
    def name(self) -> str:
        return f"Transform({self.img_generator}, {self.transform_function})"

    @property
    def parameters(self) -> torch.Tensor:
        return self.img_generator.parameters

    def get_image(self, transform=True) -> torch.Tensor:
        if transform:
            return self.transform_function(self.img_generator.get_image())
        else:
            return self.img_generator.get_image()
