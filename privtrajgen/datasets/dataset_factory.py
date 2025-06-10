#!/usr/bin/env python3
""" """

from .base_dataset import TrajectoryDataset
from .fs_nyc import FSNYCDataset
from .geolife import GeoLifeDataset
from .mnist_data import mnist_sequential
from .porto import PortoDataset
from privtrajgen.config import DatasetModes
from privtrajgen.config.enums import Datasets


def get_dataset(
        dataset_name: Datasets,
        mode: DatasetModes = DatasetModes.ALL,
        latlon_only: bool = False,
        normalize: bool = True,
        return_labels: bool = False,
        return_conditional: bool = False,
        keep_original: bool = False,
        sort: bool = False,
        mnist_dim: int = 28,
) -> TrajectoryDataset:
    args = {
        'mode': mode,
        'latlon_only': latlon_only,
        'normalize': normalize,
        'return_labels': return_labels,
        'return_conditional': return_conditional,
        'keep_original': keep_original,
        'sort': sort,
    }
    if dataset_name == Datasets.FSNYC:
        dataset: TrajectoryDataset = FSNYCDataset(**args)  # type: ignore
    elif dataset_name == Datasets.GEOLIFE:
        dataset: TrajectoryDataset = GeoLifeDataset(**args)  # type: ignore
    elif dataset_name == Datasets.PORTO:
        dataset: TrajectoryDataset = PortoDataset(**args)  # type: ignore
    elif dataset_name == Datasets.MNISTSequential:
        if not normalize or not return_labels:
            raise ValueError("MNIST dataset only supports normalize=True and return_labels=True.")
        dataset = mnist_sequential(dim=mnist_dim)
        return dataset
    else:
        raise NotImplementedError(f"Unsupported Dataset: {str(dataset_name)}")
    return dataset
