#!/usr/bin/env python3
"""Porto Taxi Dataset as PyTorch dataset.

See: https://www.kaggle.com/datasets/crailtap/taxi-trajectory
"""
import logging
import os
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
import torch

from privtrajgen import config
from privtrajgen.config import LAT, LON, SPATIAL_COLUMNS, SPATIAL_FEATURE, TIMESTAMP, TID, UID, DatasetModes, \
    Datasets
from privtrajgen.datasets.difftraj_dataset import difftraj_conditional_paths, dataset_to_difftraj_npy, difftraj_paths
from privtrajgen.utils import normalize_points
from privtrajgen.utils.helpers import printc
from .base_dataset import TrajectoryDataset
from .conditional_info import compute_conditional_info, normalise_conditional_info
from .constants import REF_POINT_PORTO, SCALE_FACTOR_PORTO, MAX_LEN_PORTO, NUM_PROCESSED_TRAJS_PORTO

# Paths
PORTO_DIR = config.BASE_DIR + 'data/porto_processed/'  # Contains one csv file per trajectory
NPY_FILE = config.BASE_DIR + 'data/porto_interpolated/porto.npy'  # Contains all trajectories as numpy array

# Constants
BBOX_PORTO = {
    'lat_min': 41.10,
    'lat_max': 41.24,
    'lon_min': -8.72,
    'lon_max': -8.50
}
MEAN_COND_PORTO = {
    'avg_dist': 115.02248445819382,
    'avg_speed': 7.6506892110659495,
    'len': 43.813086635595354,
    'total_dist': 5004.963005985399,
    'total_dur': 644.4363872963792}
STD_COND_PORTO = {
    'avg_dist': 58.176099375100776,
    'avg_speed': 3.8352678598805037,
    'len': 19.41204569359798,
    'total_dist': 3720.3542411349054,
    'total_dur': 311.1806938497962}

log = logging.getLogger()


class PortoDataset(TrajectoryDataset):

    # noinspection PyUnusedLocal
    def __init__(self,
                 mode: DatasetModes = DatasetModes.ALL,  # Ignored, only for compatibility
                 spatial_columns: list = SPATIAL_COLUMNS,
                 latlon_only: bool = True,
                 normalize: bool = True,
                 return_labels: bool = False,
                 sort: bool = False,
                 keep_original: bool = False,
                 reference_point: Union[List[float], dict] = REF_POINT_PORTO,
                 scale_factor: Union[List[float], dict] = SCALE_FACTOR_PORTO,
                 return_conditional: bool = False,
                 ):
        if not latlon_only:
            raise NotImplementedError('Only latlon data is available for Porto dataset.')
        # Call superclass / store parameters
        features = [SPATIAL_FEATURE] if latlon_only else [SPATIAL_FEATURE, 'hour', 'day']
        super().__init__(spatial_columns=spatial_columns, name=Datasets.PORTO, max_len=MAX_LEN_PORTO,
                         reference_point=reference_point, scale_factor=scale_factor, features=features,
                         latlon_only=latlon_only, sort=sort, return_labels=return_labels, normalize=normalize,
                         keep_original=keep_original, return_conditional=return_conditional)
        self.path = PORTO_DIR

        log.info(f"Reading trajectories from '{self.path}'.")
        path = Path(self.path)
        for child in path.glob("*.csv"):
            tid = child.name.replace('.csv', '')
            self.tids.append(tid)

        self.tids = sorted(self.tids)

        if NUM_PROCESSED_TRAJS_PORTO is not None:
            assert len(self.tids) == NUM_PROCESSED_TRAJS_PORTO, (f"Found {len(self.tids)} files, "
                                                                 f"expected {NUM_PROCESSED_TRAJS_PORTO}.")

    def __getitem__(self, index: int) -> Union[
        torch.Tensor, tuple[torch.Tensor, str], tuple[torch.Tensor, torch.Tensor]]:
        tid = self.tids[index]
        encoding = self.get_encoding(tid)
        if self.return_labels:
            return encoding, tid
        elif self.return_conditional:
            conditional_info = self.get_conditional_info(tid)
            return encoding, conditional_info
        else:
            return encoding

    def get_encoding(self, tid: str) -> torch.Tensor:
        self.check_tid(tid=tid)
        if tid not in self._encodings:
            # Generate encoding if not existing
            self._encode(tid)
        return self._encodings[tid]

    def get_conditional_info(self, tid: str) -> torch.Tensor:
        self.check_tid(tid=tid)
        if tid not in self._conditional_info:
            # Generate encoding if not existing
            self._encode(tid)
        return self._conditional_info[tid]

    def get_original(self, tid: str) -> pd.DataFrame:
        self.check_tid(tid=tid)
        if not self.keep_original:
            raise ValueError("Original data not kept.")
        if tid not in self._originals:
            # Generate encoding if not existing
            self._encode(tid)
        return self._originals[tid]

    def _encode(self, tid: str) -> None:
        # Load from file
        df = pd.read_csv(
            f"{self.path}/{tid}.csv",
            dtype={LAT: float,
                   LON: float,
                   TID: str,
                   UID: str,
                   },
            parse_dates=[TIMESTAMP],
            date_format="%Y-%m-%d %H:%M:%S"
        )
        if self.keep_original:
            self._originals[tid] = df
        # Convert to Tensor
        latlon = df[self.columns]
        if self.normalize:
            latlon, _, _ = normalize_points(latlon, ref=self.reference_point, scale_factor=self.scale_factor,
                                            columns=self.columns)
        latlon = torch.from_numpy(latlon.to_numpy())
        if self.latlon_only:
            self._encodings[tid] = latlon
        else:
            raise NotImplementedError('Only latlon data is available for Porto dataset.')
        # Conditional Information
        if self.return_conditional:
            cond_info = compute_conditional_info(
                traj=df, bbox=BBOX_PORTO, lat=LAT, lon=LON, categorical_timestamp=False, as_dict=True,
            )
            if self.normalize:
                cond_info = normalise_conditional_info(cond_info, MEAN_COND_PORTO, STD_COND_PORTO)  # type: ignore
            self._conditional_info[tid] = cond_info


if __name__ == '__main__':
    # Prepare the dataset: Create conditional information and save it to disk

    tmp_file = os.path.join("/tmp", "porto_tmp.npy")

    # Convert the dataset to npy
    dataset = PortoDataset(latlon_only=True, return_labels=False, keep_original=False, normalize=True,
                           return_conditional=True)
    dataset_to_difftraj_npy(dataset, tmp_file, cond_path=difftraj_conditional_paths[Datasets.PORTO])

    # Compare
    npy1 = np.load(difftraj_paths[Datasets.PORTO])
    npy2 = np.load(tmp_file)
    printc("Results are the same:", np.allclose(npy1, npy2, atol=1e-6))

    # Load the conditional npy file
    npy_cond = np.load(difftraj_conditional_paths[Datasets.PORTO])
    printc("Shape of npy conditional:", npy_cond.shape)
    printc("First Element of Conditional:", npy_cond[0])
