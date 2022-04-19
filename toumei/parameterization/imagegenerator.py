import torch
import numpy as np
import matplotlib.pyplot as plt


class ImageGenerator(object):
    def __init__(self):
        super(ImageGenerator, self).__init__()

    def __str__(self):
        return self.name

    def get_image(self, *args, **kwargs) -> torch.Tensor:
        """
        Returns an image generated using the parameters
        :param args: arguments
        :param kwargs: keyword arguments
        :return: image (tensor)
        """
        return NotImplementedError

    def numpy(self, *args, **kwargs):
        """
        Converts the generated image into a numpy array
        :param args: arguments
        :param kwargs: keyword arguments
        :return: image (numpy array)
        """
        image = self.get_image(*args, **kwargs).cpu().detach().numpy()
        image = np.transpose(image, [0, 2, 3, 1])
        return (image[0] * 255).astype(np.uint8)

    def plot_image(self):
        """
        Plots the generated image
        :return: nothing
        """
        plt.imshow(self.numpy(False))
        plt.show()

    def to(self, device: torch.device):
        return

    @property
    def name(self) -> str:
        """
        The name of the image generator
        :return: the generator name
        """
        return "Generator()"

    @property
    def parameters(self) -> torch.Tensor:
        """
        Returns the image parameters
        :return: the parameters
        """
        return NotImplementedError
