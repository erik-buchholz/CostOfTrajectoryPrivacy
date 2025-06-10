#!/usr/bin/env python3
"""Different Padding implementations for PyTorch"""
import logging
from typing import Union, List, Optional, Dict, Tuple, Sequence, Any

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


log = logging.getLogger()

def pre_pad(
        sequences: Union[Tensor, List[Tensor]],
        batch_first: bool = False,
        padding_value: float = 0.0,
) -> Tensor:
    sequences = tuple(map(lambda s: s.flip(0), sequences))
    padded_sequence = pad_sequence(sequences, batch_first, padding_value)
    _seq_dim = padded_sequence.dim()
    padded_sequence = padded_sequence.flip(-_seq_dim + batch_first)
    return padded_sequence


def pad_feature_first(batch: List[Union[torch.Tensor, List[torch.Tensor]]], padding_type: str = 'pre') -> List[torch.Tensor]:
    """
    Pads a batch of variable-length sequences to the same length.
    Considers LSTM-TrajGAN style batches where the second dimension is the number of features.

    :param batch: List of Tensors with shape
                  (batch_size, sequence_length, feature_dim) or
                  List of Lists of Tensors with shape
                  (batch_size, num_features, sequence_length, feature_dim)
                  feature_dim might depend on the feature,e.g.:
                  batch[0][0].shape = (sequence_length, 2), but
                    batch[0][1].shape = (sequence_length, 24), and
                    batch[0][2].shape = (sequence_length, 10)
    :param padding_type: Padding type ('pre' or 'post')
    :return: List[Tensor]: Padded and collated batch with shape:
             (num_features, batch_size, max_sequence_length, feature_dim)
             If the input was of the first form (without explicit num_features),
             the output shape will have num_features = 1.
             feature_dim can be different for each feature. Therefore, the output is a list and not a tensor.
    """
    log.warning("Deprecated function. Use ZeroPadding instead.")
    padded_batch =  ZeroPadding(
        return_len=False,
        return_labels=False,
        padding_value=0.0,
        padding_type=padding_type,
        fixed_length=None,
        feature_first=True,
    )(batch)['samples']

    return padded_batch


class ZeroPadding:
    def __init__(
            self,
            return_len: bool = True,
            return_labels: bool = False,
            padding_value: float = 0.0,
            padding_type: str = 'post',
            fixed_length: Optional[int] = None,
            feature_first: bool = False,
    ):
        """
        Initialize the ZeroPadding object.

        :param return_len: Whether to return a list containing the original lengths.
        :param return_labels: Whether the batch also contains labels which should be returned.
        :param padding_value: Value to use for padding.
        :param padding_type: Padding type ('pre' or 'post').
        :param fixed_length: If not None, all sequences will be padded to this length.
        :param feature_first: Whether the batch is in feature-first format.
            If true, batch has shape (num_features, batch_size, max_sequence_length, feature_dim).
            If false, batch has shape (batch_size, max_sequence_length, feature_dim).
        """
        if padding_type not in ['pre', 'post']:
            raise ValueError(f"Unknown padding type {padding_type}.")
        self.return_len = return_len
        self.return_labels = return_labels
        self.padding_value = padding_value
        self.padding_type = padding_type
        self.fixed_length = fixed_length
        self.feature_first = feature_first

    def pad(self, batch: List[Union[torch.Tensor, np.ndarray]]) -> Dict[str, Union[torch.Tensor, List]]:
        """
        Pad a batch of sequences to the same length with the specified padding value.

        :param batch:
            If self.feature_first: A list of lists with shape (batch_size, num_features, sequence_length, feature_dim).
            If not self.feature_first: A list of sequences with shape (batch_size, sequence_length, feature_dim).
        :return: The padded batch, optionally with original lengths and labels.
        """
        # Error handling for empty batch
        if not batch:
            raise ValueError("Batch is empty.")

        lengths, labels = None, None
        if self.return_labels:
            batch, labels = zip(*batch)  # type: ignore

        if (not self.feature_first or # Case 1: We add the fake feature dimension and remove it later
                (self.feature_first and (isinstance(batch[0], torch.Tensor) or isinstance(batch[0], np.ndarray))
                 and len(batch[0].shape) == 2)
                # Case 2: We want the feature dimension, but it is not present
        ):
            # Add a feature dimension to the batch to only have one padding code
            # New shape: (batch_size, 1, sequence_length, feature_dim)
            batch = [[e] for e in batch]

        # Assume all features have the same length, so we can take the length of the first feature
        lengths = [len(features[0]) for features in batch]

        # Batch has shape (batch_size, num_features, sequence_length, feature_dim): List[List[np.ndarray]]
        # Swap first two dimensions to get (num_features, batch_size, sequence_length, feature_dim): List[List[np.ndarray]]
        batch = [[torch.from_numpy(e) if isinstance(e, np.ndarray) else e for e in feature] for feature in zip(*batch)]

        # Batch has shape (num_features, batch_size, sequence_length, feature_dim): List[List[torch.Tensor]]
        if self.fixed_length is not None:
            # Pad to fixed length: Append sequence of fixed length
            batch = [feature + [torch.ones((self.fixed_length, *feature[0].shape[1:]), dtype=feature[0].dtype)]
                     for feature in batch]
        # Pad each feature separately
        padded_batch = []
        for feature in batch:
            if self.padding_type == 'pre':
                padded_feature = pre_pad(feature, batch_first=True, padding_value=self.padding_value)
            elif self.padding_type == 'post':
                padded_feature = pad_sequence(feature, batch_first=True, padding_value=self.padding_value)
            else:
                raise ValueError(f"Unknown padding type {self.padding_type}.")
            padded_batch.append(padded_feature)
        # Remove the appended sequence
        if self.fixed_length is not None:
            padded_batch = [feature[:-1] for feature in padded_batch]
        # Verify that the number of lengths is the same as the number of elements per feature
        if len(lengths) != len(padded_batch[0]):
            raise RuntimeError(
                f"Padding failed. Expected len(lengths) == {len(padded_batch[0])}, but got {len(lengths)}."
            )
        # Verify that the all features are padded to the fixed length
        if self.fixed_length and any(feature.shape[1] != self.fixed_length for feature in padded_batch):
            raise RuntimeError(
                f"Padding failed. One feature not padded to fixed length."
            )

        if not self.feature_first:
            assert len(padded_batch) == 1, "Expected only one feature."
            padded_batch = padded_batch[0]

        output = {
            'samples': padded_batch,
            'lengths': lengths if self.return_len else None,
            'labels': labels if self.return_labels else None,
        }

        return output

    def __call__(self, batch: List[Union[torch.Tensor, np.ndarray]]) -> Dict[str, Union[torch.Tensor, List]]:
        """
        Make the object callable.

        :param batch: A list of sequences (torch.Tensor or numpy.ndarray).
        :return: The padded batch, optionally with original lengths and labels.
        """
        return self.pad(batch)


class DiffTrajPadding:
    """This was supposed to be a padding implementation for the DiffTraj model.
    However, DiffTraj uses interpolation instead of padding, so this class is not used.
    """
    def __init__(self, max_length: int):
        self.max_length = max_length
        self.padding = ZeroPadding(return_len=True, return_labels=False, fixed_length=max_length)

    def __call__(self, samples: List[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        padded = self.padding(samples)
        data: torch.Tensor = padded['samples']
        lengths = padded['lengths']

        # Transpose to (batch_size, 2, traj_length)
        data = data.transpose(2, 1)

        # Attributes should be a tensor of shape (batch_size, attr_dim), where attr_dim is 1 (length) for now
        attr = torch.tensor(lengths).unsqueeze(1)
        return data, attr


def resample_trajectory(x: np.ndarray, length: int = 200) -> np.ndarray:
    """
    Resamples a trajectory to a new length.
    Source: https://github.com/Yasoz/DiffTraj/blob/main/utils/utils.py

    Parameters:
        x (np.ndarray): original trajectory, shape (N, 2)
        length (int): length of resampled trajectory

    Returns:
        np.ndarray: resampled trajectory, shape (length, 2)
    """
    len_x = len(x)
    # Time steps are evenly spaced in the original trajectory
    time_steps = np.arange(length) * (len_x - 1) / (length - 1)
    x = x.T  # Transpose to get shape (2, N)
    resampled_trajectory = np.zeros((2, length))
    for i in range(2):
        resampled_trajectory[i] = np.interp(time_steps, np.arange(len_x), x[i])
    return resampled_trajectory.T


class InterpolateTraj:
    def __init__(self, max_length: int, transpose: bool = True, conditional: bool = False):
        """
        Initialize the InterpolateTraj object.
        :param max_length: Maximum length of the resampled trajectories.
        :param transpose: Whether to transpose the trajectories to (batch_size, 2, traj_length).
        :param conditional: Whether the dataset returns conditional information.
        """
        self.max_length = max_length
        self.transpose = transpose
        self.conditional = conditional

    def __call__(self, batch: Sequence[Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Interpolates trajectories to have the same length.
        batch: List of shape (Batch_Size, Length, 2) if not conditional, else List of tuples (traj, attr), where
                traj is of shape (Length, 2) and attr is a tensor of shape (8).
        """
        # If self.conditional, each element in the batch is a tuple (traj, attr)
        attr: torch.Tensor
        if self.conditional:
            batch, attr = zip(*batch)
        else:
            # Attr is lengths of trajectories
            attr = torch.tensor([len(traj) for traj in batch]).unsqueeze(1)
        # Assert shape of attributes
        if self.conditional:
            assert attr.shape == (len(batch), 8), f"Expected shape (batch_size, 8), got {attr.shape}."
        else:
            assert attr.shape == (len(batch), 1), f"Expected shape (batch_size, 1), got {attr.shape}."
        # Interpolate trajectories
        batch = torch.tensor(np.array([resample_trajectory(traj, self.max_length) for traj in batch]))
        # Convert to tensor and transpose to (batch_size, 2, traj_length)
        if self.transpose:
            batch = batch.transpose(1, 2)
        return batch, attr