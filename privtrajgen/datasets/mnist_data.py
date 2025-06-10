#!/usr/bin/env python3
"""
Creates MNIST sequential as PyTorch dataset with variable format.
This dataset is used by RGAN as default dataset.
"""
from typing import Optional, Tuple

import torch
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from torchvision.utils import make_grid

from privtrajgen import config
from privtrajgen.config import MNISTSequential, FeatureType
from privtrajgen.datasets import TrajectoryDataset


class Reshape:

    def __init__(self, size: Tuple[int, int]):
        self.shape = size

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.reshape(tensor, shape=self.shape)


def mnist_sequential(dim: int = 28) -> TrajectoryDataset:
    if not isinstance(dim, int):
        raise ValueError(f"dim must be int, got {type(dim)}")
    mnist = datasets.MNIST(
        root=config.BASE_DIR + '/data/mnist', train=True, download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5),
                                      Reshape(size=(28 * 28 // dim, dim))]))
    # Make this a TrajectoryDataset
    mnist.name = MNISTSequential
    mnist.features = ['mnist']
    mnist.vocab_size = {'mnist': dim}
    mnist.embedding_size = {'mnist': dim}
    mnist.feature_types = {'mnist': FeatureType.CONTINUOUS}
    return mnist


def show_mnist_samples(samples: torch.Tensor, nrow: int = 5, ax: Optional[plt.Axes] = None,
                       fig: Optional[plt.Figure] = None) -> Tuple[Optional[plt.Figure], plt.Axes]:
    grid = make_grid(samples, nrow=nrow, normalize=True).permute(1, 2, 0).detach().cpu().numpy()
    # Display the grid of images
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(grid)
    ax.axis('off')
    return fig, ax
