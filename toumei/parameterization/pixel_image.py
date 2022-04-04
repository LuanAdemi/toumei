import torch

from toumei.parameterization import Generator


class PixelImage(Generator):
    """
    A basic pixel based image generator.
    This generator exposes the raw pixel values as parameters to the optimizer.
    Compared to other parameterization approaches it performs pretty bad.
    """
    def __init__(self, *dims: int):
        super(PixelImage, self).__init__()
        self.image = torch.rand(tuple(dims), requires_grad=True)

    @property
    def parameters(self) -> list:
        return [self.image]

    def get_image(self, *args, **kwargs) -> torch.Tensor:
        return self.image
