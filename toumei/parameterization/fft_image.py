import torch

from toumei.parameterization import ImageGenerator


class FFTImage(ImageGenerator):
    """
    A fast fourier parameterized image
    This generator exposes the DFT values as parameters to the optimizer.
    """
    def __init__(self, *shape: int, saturation: float = 4.0):
        """
        Initializes a new fast fourier transformation image generator.
        This encodes images in the spatial domain.

        :param shape: the shape of the image
        :param saturation: this controls the saturation of the images
        """
        super(FFTImage, self).__init__()
        self.shape = shape
        self.saturation = saturation
        self.w = shape[-2]
        self.h = shape[-1]
        fy = torch.fft.fftfreq(self.h)[:, None]

        if self.w % 2 == 1:
            fx = torch.fft.fftfreq(self.w)[: self.w // 2 + 2]
        else:
            fx = torch.fft.fftfreq(self.w)[: self.w // 2 + 1]

        self.freq = torch.sqrt(fx ** 2 + fy ** 2)
        self.spectrum = (torch.randn(*self.shape[:2] + self.freq.shape + (2,)) * 0.01).requires_grad_(True)

    def freq_to_img(self, frequency):
        return torch.fft.irfftn(torch.view_as_complex(frequency), s=(self.h, self.w), norm="ortho")

    @property
    def name(self) -> str:
        return f"FFTImage({self.shape})"

    @property
    def parameters(self) -> list:
        return [self.spectrum]

    def get_image(self,  *args, **kwargs) -> torch.Tensor:
        scale = 1.0 / torch.maximum(self.freq, torch.tensor(1.0 / max(self.w, self.h)))[None, None, ..., None]
        scaled_spectrum = scale * self.spectrum
        image = self.freq_to_img(scaled_spectrum)[:self.shape[0], :self.shape[1], :self.h, :self.w] / self.saturation
        return torch.sigmoid(image)
