#!/usr/bin/env python3
"""This module contains wrappers for the datasets and data loaders."""
from typing import List, TypeVar, Sequence, cast, Dict

import numpy as np
import pandas as pd

from privtrajgen.config import GEOLIFE, PORTO, MNISTSequential, Datasets, SPATIAL_COLUMNS
from privtrajgen.config.enums import FSNYC
from privtrajgen.utils import denormalise
from .base_dataset import TrajectoryDataset
from .constants import *
from .constants import ref_points, scale_factors
from .dataset_factory import get_dataset
from .difftraj_dataset import get_difftraj_dataset
from .fs_nyc import FSNYCDataset, BBOX_FS
from .geolife import (GeoLifeDataset, DEFAULT_BBOX_GEOLIFE)
from .mnist_data import mnist_sequential
from .padding import ZeroPadding, DiffTrajPadding
from .porto import PortoDataset, BBOX_PORTO
from .split_dataset import random_split
from .utils import DictProperty

DATASET_CLASSES = {
    FSNYC: FSNYCDataset,
    GEOLIFE: GeoLifeDataset,
    MNISTSequential: mnist_sequential,
    PORTO: PortoDataset,
}

dataset_bounding_boxes = {
    FSNYC: BBOX_FS,
    GEOLIFE: DEFAULT_BBOX_GEOLIFE,
    PORTO: BBOX_PORTO,
}

T = TypeVar("T", pd.DataFrame, np.ndarray, Sequence[np.ndarray])


def denormalise_by_dataset(data: T, dataset: Datasets,
                           column_order: List[str] = SPATIAL_COLUMNS) -> T:
    """Denormalise data from a specified dataset.
    :param data: Dataset to denormalise. Shape can be:
        - (n_trajectories, n_points, n_features) for trajectories
        - (n_points, n_features) for points
        - List[np.ndarray], each element with shape (n_points, n_features)
        - pd.DataFrame with columns column names as provided in 'columns'
    :param dataset: Dataset name
    :param column_order: List of column names
    """
    # cast ref and scale for mypy
    ref = cast(Dict[int | str, float], ref_points[dataset])
    scale = cast(Dict[int | str, float], scale_factors[dataset])
    return denormalise(data, ref=ref, scale=scale, columns=column_order)
