#!/usr/bin/env python3
"""Provides diverse helper functions."""
import logging
import random
import re
from collections import OrderedDict
from dataclasses import field
from datetime import timedelta
from types import SimpleNamespace
from typing import Union, Tuple, Sequence, Any, Optional, Callable

import math
import numpy as np
import pandas as pd
import scipy
import torch.nn

from privtrajgen.config import LAT, LON, TID

log = logging.getLogger()


def logc(header: str, *args: Any, column_width: int = 40, level: int = logging.INFO) -> None:
    txt = f"{header:<{column_width}}" + "".join([str(arg) for arg in args])
    log.log(level, txt)


# Create a print function that prints with one header columns and one content column
def printc(header: str, *args: Any, column_width: int = 40) -> None:
    """
    Print first argument in the first column and the rest in the second column.
    Example:
        printc("Name", "Value") --> "Name:       Value"
    :param header: Header to print
    :param args: Arguments to print
    :param column_width: Width of the first column
    :return:
    """
    print(f"{header:<{column_width}}", *args)


def clean_filename(filename: str) -> str:
    """Remove special characters from a filename."""
    return re.sub(r'[^\w.-]', '_', filename).strip().replace(' ', '_')


def time_to_str(time: Union[float, timedelta, int]) -> str:
    """Convert time in seconds to a human-readable format.
    XXs for seconds, XX:XX for minutes and seconds, XX:XX:XX for hours, minutes and seconds.

    :param time: Time in seconds
    :return: Time in a human-readable format
    """
    # Convert timedelta to seconds
    if isinstance(time, timedelta):
        time = time.total_seconds()
    if time < 60:
        return f"{time:.2f}s"
    elif time < 3600:
        return f"{int(time // 60):02d}:{int(time % 60):02d}min"
    else:
        return f"{int(time // 3600):02d}:{int((time % 3600) // 60):02d}:{int(time % 60):02d}h"


def set_random_seed(seed: int) -> None:
    """Set the random seed in our used libraries."""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    logc("Random seed set to: ", seed)


def compute_inverse_power(x: float) -> int:
    """Compute y such that 10^-y <= x <= 10^-(y-1)"""
    return math.ceil(-math.log10(x))


# Differential Privacy Helper Code
def compute_delta(n_train: int) -> float:
    """Compute delta such that delta = 10^-x <= 1/n_train.
    Setting delta to the inverse of the size of the training set is
    the best practice in literature.
    (e.g., https://www.cleverhans.io/privacy/2019/03/26/machine-learning-with-differential-privacy-in-tensorflow.html)
    Updated recommendation of 1/n^{1.1} according to
    [1] N. Ponomareva et al.,
    “How to DP-fy ML: A Practical Guide to Machine Learning with Differential Privacy.”
    arXiv, Mar. 02, 2023. doi: 10.48550/arXiv.2303.00654.
    """
    # x = compute_inverse_power(1/n_train)
    # delta = 10 ** (-x)
    delta = 1 / (n_train ** 1.1)
    return delta


def count_parameters_torch(model: torch.nn.Module, print_layers: bool = True) -> int:
    """
    Compute the total number of parameters.
    :param model: PyTorch model
    :param print_layers: Print the parameters in each layer (Weights + Bias)
    :return: Total number of parameters
    """
    total = 0
    layers: OrderedDict[str, int] = OrderedDict()
    for name, p in model.named_parameters():
        parts = name.split('.')
        name = parts[0] + '.' + parts[1]
        if name in layers:
            layers[name] += p.numel()
        else:
            layers[name] = p.numel()
        total += p.numel()
    if print_layers:
        for layer in layers:
            log.info(f"{layer}: {layers[layer]}")
    return total


def find_bbox(
        df: pd.DataFrame,
        quantile: float = 1,
        x_label: str = LON,
        y_label: str = LAT
) -> Tuple[float, float, float, float]:
    """Find a bounding box enclosing the defined quantile of points.

    :return: (Minimum X, Maximum X, Minimum Y, Minimum Y)
    """
    upper_quantiles = df.quantile(q=quantile, numeric_only=True)
    lower_quantiles = df.quantile(q=(1 - quantile), numeric_only=True)
    return lower_quantiles[x_label], upper_quantiles[x_label], \
        lower_quantiles[y_label], upper_quantiles[y_label]


def get_ref_point(series: Union[np.ndarray, pd.Series]) -> Union[float, np.ndarray]:
    """
    Get the reference point for normalization.
    :param series: Series or array of points
    :return: Reference point
    """
    # Convert to numpy array
    if type(series) is pd.Series or type(series) is pd.DataFrame:
        series = series.to_numpy()
    if len(series.shape) == 2:
        # nD points
        ref_point = (np.max(series, axis=0) + np.min(series, axis=0)) / 2
    else:
        # 1D points
        ref_point = (max(series) + min(series)) / 2
    return ref_point


def get_scaling_factor(series: Union[np.ndarray, pd.Series], ref: Union[float, Sequence[float], np.ndarray]) -> Union[
    float, np.ndarray]:
    """
    Get the scale factor for normalization.
    :param series: Series or array of points
    :param ref: Reference point
    :return: Scale factor
    """
    if type(series) is pd.Series or type(series) is pd.DataFrame:
        series = series.to_numpy()
    if len(series.shape) == 2:
        # nD points
        scale_factor = np.max(abs(series - ref), axis=0)
    else:
        # 1D points
        scale_factor = max(abs(series - ref))
    return scale_factor


def dict2mdtable(d: dict, key: str = 'Name', val: str = 'Value') -> str:
    """
    Turn a dictionary into a Markdown table. Can be used to write hyperparameters into TensorBoard.
    Source: https://github.com/tensorflow/tensorboard/issues/46#issuecomment-1331147757
    :param d: Dictionary
    :param key: Title of Key column
    :param val: Title of Value colum
    :return: Markdown table as str
    """
    rows = [f'| {key} | {val} |']
    rows += ['|--|--|']
    rows += [f'| {k} | {v} |' for k, v in d.items()]
    return "  \n".join(rows)


def df2trajectory_dict(df: pd.DataFrame, tid_label: str = TID) -> dict:
    result = {tid: df for tid, df in df.groupby(tid_label)}
    return result


def dict_to_namespace(d: dict) -> SimpleNamespace:
    if isinstance(d, SimpleNamespace):
        return d
    for key, value in d.items():
        d[key] = SimpleNamespace(**value) if isinstance(value, dict) else value
    return SimpleNamespace(**d)


def mean_confidence_interval(
    data: Union[Sequence[float], np.ndarray], confidence: float = 0.95
) -> Tuple[float, float, float]:
    """Calculate the mean and confidence interval of a list of numbers.

    Source: https://stackoverflow.com/a/15034143/18265340

    :param data: List of numbers
    :param confidence: Confidence level
    :return: Mean, Lower Bound, Upper Bound
    """
    if len(data) == 0:
        raise ValueError("The data list is empty.")
    if not 0 < confidence < 1:
        raise ValueError("Confidence level must be between 0 and 1.")
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m - h, m + h


def mean_conf_half(
        data: Union[Sequence[float], np.ndarray], confidence: float = 0.95
) -> Tuple[float, float]:
    """Calculate the mean and half-width of the confidence interval for a list of numbers.

    Source: https://stackoverflow.com/a/15034143/18265340

    :param data: List of numbers
    :param confidence: Confidence level
    :return: Mean and half-width of the confidence interval
    """
    if len(data) == 0:
        raise ValueError("The data list is empty.")
    if not 0 < confidence < 1:
        raise ValueError("Confidence level must be between 0 and 1.")
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, h


def enum_field(default: Optional[Any] = None, default_factory: Optional[Callable] = None, *args: Any,
               **kwargs: Any) -> Any:
    if default:
        kwargs["default"] = default
    if default_factory:
        kwargs["default_factory"] = default_factory
    kwargs["metadata"] = {"custom_args": {"metavar": None}}
    return field(*args, **kwargs)
