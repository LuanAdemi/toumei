from toumei.cnns.featurevis.parameterization.imagegenerator import ImageGenerator
import torch
import torchvision.transforms as T

standard_transform = T.Compose([
    T.Pad(12),
    T.RandomRotation((-10, 11))
])


class Transform(ImageGenerator):
    """
    A wrapper for generators for applying torchvision transforms on the given image generator.
    """
    def __init__(self, generator: ImageGenerator, transform_func=standard_transform):
        """
        Creates a new Transform wrapper

        :param generator: the base image generator
        :param transform_func: the transform function
        """
        super(Transform, self).__init__()
        self.img_generator = generator
        self.transform_function = transform_func

    def to(self, device: torch.device):
        self.img_generator.to(device)

    @property
    def name(self) -> str:
        return self.img_generator.name

    @property
    def parameters(self) -> torch.Tensor:
        return self.img_generator.parameters

    def get_image(self, transform=True) -> torch.Tensor:
        img = self.img_generator.get_image()
        if transform:
            img = self.transform_function(img)

        return img
