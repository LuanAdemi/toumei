import torch

from toumei.parameterization import ImageGenerator


class PixelImage(ImageGenerator):
    """
    A basic pixel based image generator.
    This generator exposes the raw pixel values as parameters to the optimizer.
    Compared to other parameterization approaches it performs pretty bad.
    """
    def __init__(self, *shape: int):
        super(PixelImage, self).__init__()
        self.image = torch.rand(tuple(shape), requires_grad=True)
        self.shape = shape

    @property
    def name(self) -> str:
        return f"PixelImage({self.shape})"

    @property
    def parameters(self) -> list:
        return [self.image]

    def get_image(self, *args, **kwargs) -> torch.Tensor:
        return self.image
