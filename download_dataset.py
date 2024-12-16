import os

from datasets import load_dataset
from torch.utils.data import DataLoader


local_dir = os.path.dirname(os.path.realpath(__file__))
dataset_path = os.path.join(local_dir, "dataset")

ds = load_dataset("Artificio/WikiArt_Full", cache_dir=dataset_path).with_format("torch")

dataloader = DataLoader(ds["train"], batch_size=4)

for batch in dataloader:
    print(batch)
    break

pass

# https://huggingface.co/datasets/Artificio/WikiArt_Full
# https://huggingface.co/docs/datasets/use_with_pytorch
