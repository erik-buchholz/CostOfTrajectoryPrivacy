#!/usr/bin/env python3
""" """
import random
import typing
from typing import Sequence, List, Dict, Union, Any

import pandas as pd
import torch

from privtrajgen.datasets import TrajectoryDataset


class SubTrajectoryDataset(TrajectoryDataset):
    dataset: TrajectoryDataset
    indices: Sequence[int]
    tids: List[str]

    def __init__(self, dataset: TrajectoryDataset, indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = indices
        self.tids = [dataset.tids[idx] for idx in indices]
        self._uids: Dict[str, Union[int, str]] = ProxyAttribute(self, '_uids')  # type: ignore
        self._encodings: Dict[str, torch.Tensor] = ProxyAttribute(self, '_encodings')  # type: ignore
        self._originals: Dict[str, pd.DataFrame] = ProxyAttribute(self, '_originals')  # type: ignore

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: Union[int, List[int]]) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
        if isinstance(idx, list):
            return [self.dataset[self.indices[i]] for i in idx]
        return self.dataset[self.indices[idx]]

    # Pass through all other methods except the defined ones, uid, encoding and original
    def __getattr__(self, item: str) -> torch.Tensor:
        """Return the attribute from the dataset. For uid, encoding and original, only return value if accessed index is in self.tids or raise ValueError."""
        return getattr(self.dataset, item)

    def get_encoding(self, tid: str) -> torch.Tensor:
        if tid not in self.tids:
            raise ValueError(f"Trajectory ID '{tid}' not in dataset.")
        return self.dataset.get_encoding(tid)

    def get_conditional_info(self, tid: str) -> torch.Tensor:
        if tid not in self.tids:
            raise ValueError(f"Trajectory ID '{tid}' not in dataset.")
        return self.dataset.get_conditional_info(tid)

    def get_original(self, tid: str) -> pd.DataFrame:
        if tid not in self.tids:
            raise ValueError(f"Trajectory ID '{tid}' not in dataset.")
        return self.dataset.get_original(tid)


class ProxyAttribute:
    def __init__(self, dataset: SubTrajectoryDataset, attr: str) -> None:
        self.parent = dataset
        self.attr = attr

    def __getitem__(self, tid: str) -> Any:
        if tid not in self.parent.tids:
            raise ValueError(f"Trajectory ID '{tid}' not in dataset.")
        return self.parent.dataset.__getattribute__(self.attr)[tid]

    def values(self) -> List[Any]:
        return [self.parent.dataset.__getattribute__(self.attr)[tid] for tid in self.parent.tids]


def random_split(dataset: TrajectoryDataset, sizes: List[Union[float, int]]) -> List[TrajectoryDataset]:
    """
    Split the dataset based on provided sizes or fractions.

    :param dataset: The dataset to split.
    :param sizes: A list of dataset sizes (as integers) or fractions (as floats).
    Returns:
        A list of SubTrajectoryDataset instances.
    """
    all_indices = list(range(len(dataset)))

    # Determine if sizes represent absolute lengths or fractions
    if all(isinstance(size, int) for size in sizes):
        assert sum(sizes) == len(dataset), "The sum of sizes should be equal to the total dataset size."
        split_sizes: List[int] = typing.cast(List[int], sizes)
    elif all(isinstance(size, float) for size in sizes):
        assert abs(sum(sizes) - 1.0) < 1e-6, "The sum of fractions should be approximately 1."
        split_sizes = [int(len(dataset) * size) for size in sizes[:-1]]  # everything except the last size
        split_sizes.append(len(dataset) - sum(split_sizes))  # remaining data for the last size
    else:
        raise ValueError("Sizes should either be all integers or all floats.")

    subsets = []
    for split_size in split_sizes:
        split_indices = random.sample(all_indices, split_size)
        all_indices = [i for i in all_indices if i not in split_indices]
        subsets.append(SubTrajectoryDataset(dataset, split_indices))

    return subsets
