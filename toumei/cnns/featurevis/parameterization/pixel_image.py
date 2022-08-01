import torch

from toumei.cnns.featurevis.parameterization.imagegenerator import ImageGenerator


class PixelImage(ImageGenerator):
    """
    A basic pixel based image generator.
    This generator exposes the raw pixel values as parameters to the optimizer.
    Compared to other parameterization approaches it performs pretty bad.
    """
    def __init__(self, *shape: int):
        """
        Initializes a new pixel based image generator.
        This optimizes the pixel values directly, which is pretty much a naive approach and performs as such.

        :param shape: The shape of the image
        """
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
