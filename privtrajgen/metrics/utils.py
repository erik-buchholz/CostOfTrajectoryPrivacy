#!/usr/bin/env python3
"""Utilities for the metrics module."""
from typing import Sequence, Union

import numpy as np


def verify_batch_sizes(batch1: Union[Sequence[np.ndarray], np.ndarray],
                       batch2: Union[Sequence[np.ndarray], np.ndarray],
                       ignore_check: bool = False) -> None:
    """Verify that the batch sizes of samples are equal.
        While this is not a strict requirement, we want to ensure that the samples are consistent for evaluation.
        This is a soundness check for the evaluation of the metrics.

        :param batch1: Batch1 (n_samples, *)
        :param batch2: Batch2 (n_samples, *)
        :param ignore_check: Ignore the check for batch sizes (for testing purposes)
    """
    if not ignore_check and len(batch1) != len(batch2):
        raise ValueError("Batch sizes of samples should be equal.")
