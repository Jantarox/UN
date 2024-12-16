import os

from datasets import load_dataset
import matplotlib.pyplot as plt

import umap

local_dir = os.path.dirname(os.path.realpath(__file__))
dataset_path = os.path.join(local_dir, "dataset")

ds = load_dataset("Artificio/WikiArt_Full", cache_dir=dataset_path).with_format("torch")

# ds["train"]["resnet50_non_robust_feats"]
# ds["train"]["resnet50_robust_feats"]
# ds["train"]["embeddings_pca512"]


for features_name in [
    "resnet50_non_robust_feats",
    "resnet50_robust_feats",
    "embeddings_pca512",
]:
    features = ds["train"][features_name]

    embedding = umap.UMAP(random_state=42).fit_transform(features)
    plt.scatter(embedding[:, 0], embedding[:, 1], s=0.1, cmap="Spectral")

    # save figure
    plt.savefig(f"umap_{features_name}.png")
    plt.clf()
