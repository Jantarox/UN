import torch

import numpy as np
from skimage.measure import grid_points_in_poly

from bezier_curves import get_random_points, get_bezier_curve

from tqdm import tqdm


def generate_damage_mask(shape=(256, 256), channels=3, fraction=1 / 16) -> torch.Tensor:
    n = 7
    bezier_points = 5
    rad = 0.4
    scale = shape[0] * np.sqrt(fraction)

    points = get_random_points(n=n, scale=scale)
    offset = np.random.rand(1, 2) * (shape[0] - scale)

    points += offset
    curve_verts, _ = get_bezier_curve(points, rad=rad, numpoints=bezier_points)
    mask = grid_points_in_poly((256, 256), curve_verts)
    mask = np.array([mask] * channels)

    if mask.sum() / mask.size > fraction:
        # print("retrying")
        return generate_damage_mask(shape=shape, channels=channels, fraction=fraction)
    return mask


def damage_image(image: torch.Tensor) -> torch.Tensor:
    # image.shape = (3, 256, 256)
    mask = generate_damage_mask(shape=image.shape[1:], channels=image.shape[0])
    image[mask] = 0
    return image


def test_damage_fraction():
    def generator():
        while True:
            yield

    shape = (256, 256)
    channels = 1
    fraction = 1 / 16
    for _ in tqdm(generator()):
        mask = generate_damage_mask(shape=shape, channels=channels, fraction=fraction)
        assert mask.sum() / mask.size <= fraction


if __name__ == "__main__":
    test_damage_fraction()
    # test results:
    #   fraction=1/4:  8/133_892 iterations had to be repeated
    #   fraction=1/16: 8/156_065 iterations had to be repeated
