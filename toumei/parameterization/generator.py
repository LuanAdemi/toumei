import torch
import numpy as np
import matplotlib.pyplot as plt


class Generator(object):
    def __init__(self):
        super(Generator, self).__init__()

    def get_image(self, *args, **kwargs) -> torch.Tensor:
        return NotImplementedError

    def numpy(self, *args, **kwargs):
        image = self.get_image(*args, **kwargs).cpu().detach().numpy()
        image = np.transpose(image, [0, 2, 3, 1])
        return image[0]

    def plot_image(self):
        plt.imshow(self.numpy(False))
        plt.show()

    @property
    def name(self) -> str:
        return "Generator()"

    @property
    def parameters(self) -> torch.Tensor:
        return NotImplementedError
