"""Low-light simulation and basic image preprocessing."""
import numpy as np
from PIL import Image

from . import config as C


def simulate_low_light(image, brightness_factor=C.LOW_LIGHT_BRIGHTNESS, gamma=C.LOW_LIGHT_GAMMA,
                       noise_std=C.LOW_LIGHT_NOISE_STD, rng=None):
    """uint8 RGB -> uint8 low-light RGB.

    1. scale to [0,1]  2. multiply by ``brightness_factor``  3. gamma curve
    4. additive Gaussian sensor noise  5. clip and return uint8.
    """
    rng = rng if rng is not None else np.random
    img = image.astype(np.float32) / 255.0
    img = img * brightness_factor
    img = np.power(np.clip(img, 0.0, 1.0), gamma)
    img = img + rng.normal(0, noise_std, img.shape).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)
    return (img * 255).astype(np.uint8)


def resize_image(img, size=C.IMAGE_SIZE):
    return img.resize((size, size), Image.Resampling.BILINEAR)


def resize_and_binarize_mask(mask, size=C.IMAGE_SIZE):
    mask = mask.convert("L").resize((size, size), Image.Resampling.NEAREST)
    arr = (np.array(mask) > 0).astype(np.uint8) * 255
    return Image.fromarray(arr)
