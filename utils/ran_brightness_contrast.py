from albumentations import Compose, RandomBrightnessContrast
import random
class RandomBrightnessContrast_corrected(RandomBrightnessContrast):
    """Randomly change brightness and contrast of the input image.

    Args:
        brightness_limit ((float, float) or float): factor range for changing brightness.
            If limit is a single float, the range will be (-limit, limit). Default: (-0.2, 0.2).
        contrast_limit ((float, float) or float): factor range for changing contrast.
            If limit is a single float, the range will be (-limit, limit). Default: (-0.2, 0.2).
        brightness_by_max (Boolean): If True adjust contrast by image dtype maximum,
            else adjust contrast by image mean.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, brightness_limit=[-0.1, 0.1], contrast_limit=[-0.2, 0.6], brightness_by_max=True, always_apply=False, p=0.5):
        super(RandomBrightnessContrast_corrected, self).__init__(always_apply, p)
        self.brightness_limit = tuple(brightness_limit)
        self.contrast_limit = tuple(contrast_limit)
        self.brightness_by_max = brightness_by_max

    def get_params(self):
        return {
            "alpha": 1.0 + random.uniform(self.contrast_limit[0], self.contrast_limit[1]),
            "beta": 0.0 + random.uniform(self.brightness_limit[0], self.brightness_limit[1]),
        }
