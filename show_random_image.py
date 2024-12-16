import os

from datasets import load_dataset
from torch.utils.data import DataLoader
import torch

import matplotlib.pyplot as plt
from image_damage import damage_image


local_dir = os.path.dirname(os.path.realpath(__file__))
dataset_path = os.path.join(local_dir, "dataset")

ds = load_dataset(
    "Artificio/WikiArt_Full", cache_dir=dataset_path, streaming=True, split="train"
).with_format("torch")

dataloader = DataLoader(ds, batch_size=4)

for batch in dataloader:
    images = batch["image"]
    images: torch.Tensor
    print(images.shape)
    # show 4 images on 1 plot
    fig, axs = plt.subplots(2, 2, squeeze=True)
    axs = axs.flatten()

    for image, ax in zip(images, axs):  # iterate over images:
        ax: plt.Axes
        image = damage_image(image)
        image.transpose_(0, 1)
        image.transpose_(1, 2)
        ax.imshow(image)

    plt.show()
    break
