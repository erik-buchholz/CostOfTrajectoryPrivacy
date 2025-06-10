#!/usr/bin/env python3
"""Base Dataset Class for Interface Definition"""
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Union, Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset

from privtrajgen.config import SPATIAL_FEATURE, DatasetModes, FeatureType, Datasets
from .constants import VOCAB_SIZE, EMBEDDING_SIZE, FEATURE_TYPES


class TrajectoryDataset(Dataset, ABC):

    @abstractmethod
    def __init__(self, spatial_columns: list, name: Datasets, max_len: int,
                 reference_point: Union[Sequence[float], Dict[str, float], Dict[int, float]],
                 scale_factor: Union[Sequence[float], Dict[str, float], Dict[int, float]],
                 mode: DatasetModes = DatasetModes.ALL, features: List[str] = [SPATIAL_FEATURE], latlon_only: bool = True,
                 sort: bool = False, return_labels: bool = True, normalize: bool = True, path: Optional[str] = None,
                 keep_original: bool = False, vocab_size: Dict[str, int] = VOCAB_SIZE,
                 embedding_size: Dict[str, int] = EMBEDDING_SIZE, feature_types: Dict[str, FeatureType] = FEATURE_TYPES,
                 return_conditional: bool = False):
        """
        Initialize the dataset.
        :param mode: Dataset mode
        :param name: Name of the dataset
        :param max_len: Maximum length of a trajectory (required by some models)
        :param path: Absolute path to the dataset file(s). Either csv file or directory containing files.
                     Overwrites Mode.
        :param spatial_columns: List of lat/lon(/alt) or x/y/z columns to read in order
        :param features: List of features to use; spatial features are always called 'latlon'.
        :param latlon_only: Only consider spatial information
        :param normalize: Normalize values to [-1;1]
        :param return_labels: Return TIDs as labels
        :param sort: Order the encodings in increasing order of sequence length to reduce required padding
        :param keep_original: Keep the original data in memory for testing
        :param reference_point: Reference point(s) for normalization. Required if normalize=True.
        :param scale_factor: Scale factor(s) for normalization. Required if normalize=True.
        :param vocab_size: Vocabulary size for each feature
        :param embedding_size: Embedding size for each feature
        :param feature_types: Feature types for each feature
        :param return_conditional: Return conditional information [used by DiffTraj]
        """
        super().__init__()
        # Store parameters
        self.name = name
        self.mode = DatasetModes.PATH if path is not None else mode
        self.path = path
        self.columns = spatial_columns
        self.features = features
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.feature_types = feature_types
        self.latlon_only = latlon_only
        # Raise value error if latlon_only true and features contains non-spatial features
        if self.latlon_only and any(f not in [SPATIAL_FEATURE] for f in self.features):
            raise ValueError("Cannot use non-spatial features if latlon_only is True.")
        self.normalize = normalize
        # Either labels or conditional information can be returned, not both
        if return_labels and return_conditional:
            raise ValueError("Cannot return both labels and conditional information.")
        self.return_labels = return_labels
        self.return_conditional = return_conditional
        self.keep_original = keep_original
        self.max_len = max_len
        self.sort = sort
        if isinstance(reference_point, dict):
            self.reference_point: Sequence[float] = [reference_point[c] for c in self.columns]
        else:
            self.reference_point = reference_point
        if isinstance(scale_factor, dict):
            self.scale_factor: Optional[Sequence[float]] = [scale_factor[c] for c in self.columns]
        else:
            self.scale_factor = scale_factor

        # Initialize Lookup Tables
        self.tids: List[str] = []  # List of trajectory IDs
        self._uids: Dict[str, Union[int, str]] = {}  # TID --> UID
        self._encodings: Dict[str, torch.Tensor] = {}  # TID --> Encoding
        self._originals: Dict[str, pd.DataFrame] = {}  # TID --> Original Dataframe
        self._conditional_info: Dict[str, torch.Tensor] = {}  # TID --> Conditional Information

    def __len__(self) -> int:
        """Return the number of trajectories in the dataset."""
        return len(self.tids)

    @abstractmethod
    def __getitem__(self, index: int) -> torch.Tensor:
        """Return the trajectory at the given index."""
        # Must be implemented by subclass
        pass

    @abstractmethod
    def get_encoding(self, tid: str) -> torch.Tensor:
        """Return the encoding for the given trajectory ID."""
        # Must be implemented by subclass
        pass

    @abstractmethod
    def get_conditional_info(self, tid: str) -> torch.Tensor:
        """Return the conditional information for the given trajectory ID."""
        # Must be implemented by subclass
        pass

    @abstractmethod
    def get_original(self, tid: str) -> pd.DataFrame:
        """Return the original data for the given trajectory ID."""
        # Must be implemented by subclass
        pass

    def check_tid(self, tid: str) -> None:
        """Check if the trajectory ID is in the dataset."""
        if tid not in self.tids:
            raise ValueError(f"Trajectory ID '{tid}' not in dataset.")
