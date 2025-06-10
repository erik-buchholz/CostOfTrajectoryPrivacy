#!/usr/bin/env python3
"""Provides access to the Foursquare NYC dataset as PyTorch dataset."""
import logging
import os
from typing import Optional, Union, Sequence, Dict, cast

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence

from privtrajgen import config
from privtrajgen.config import DatasetModes, Datasets
from privtrajgen.config.constants import *
from privtrajgen.utils import normalize_points
from .base_dataset import TrajectoryDataset
from .conditional_info import compute_conditional_info, normalise_conditional_info
from .constants import MAX_LEN_FS, REF_FS, SCALE_FS

# Paths
PATH_ALL = os.path.join(config.BASE_DIR, 'data/fs_nyc/all_latlon.csv')
PATH_TRAIN = os.path.join(config.BASE_DIR, 'data/fs_nyc/train_latlon.csv')
PATH_TEST = os.path.join(config.BASE_DIR, 'data/fs_nyc/test_latlon.csv')

# For conditional information
BBOX_FS = {
    'lat_min': 40.52,
    'lat_max': 41,
    'lon_min': -74.3,
    'lon_max': -73.65
}
MEAN_COND_FS = {
    'avg_dist': 3875.0833151958454,
    'avg_speed': 0.16955446711823782,
    'len': 21.747970120168887,
    'total_dist': 72845.96751040735,
    'total_dur': 460788.3078921728}
STD_COND_FS = {
    'avg_dist': 2711.7836210749056,
    'avg_speed': 0.24872947621866373,
    'len': 14.462239021775982,
    'total_dist': 59213.48275959241,
    'total_dur': 109376.27729735745}

log = logging.getLogger()


class FSNYCDataset(TrajectoryDataset):

    def __init__(
            self,
            mode: DatasetModes = DatasetModes.ALL,
            spatial_columns: list = SPATIAL_COLUMNS,
            latlon_only: bool = True,
            normalize: bool = True,
            return_labels: bool = False,
            sort: bool = False,
            padding: bool = False,
            path: Optional[str] = None,
            keep_original: bool = False,
            reference_point: Union[Sequence[float], Dict[str, float], Dict[int, float]] = REF_FS,
            scale_factor: Union[Sequence[float], Dict[str, float], Dict[int, float]] = SCALE_FS,
            return_conditional: bool = False,
    ):
        """
        Initialize the dataset.
        :param mode: Dataset mode
        :param path: Absolute path to the dataset file(s). Either csv file or directory containing files.
        :param spatial_columns: List of columns to read in order
        :param latlon_only: Only consider spatial information
        :param normalize: Normalize values to [-1;1]
        :param return_labels: Return TIDs as labels
        :param sort: Order the encodings in increasing order of sequence length to reduce required padding
        :param padding:
        :param keep_original: Keep the original data in memory for testing
        :param reference_point: Reference point(s) for normalization (in order of spatial_columns)
        :param scale_factor: Scale factor(s) for normalization (in order of spatial_columns)
        :param return_conditional: Return conditional information
        """
        # Call superclass
        super().__init__(spatial_columns=spatial_columns, name=Datasets.FSNYC, max_len=MAX_LEN_FS,
                         reference_point=reference_point, scale_factor=scale_factor, mode=mode, latlon_only=latlon_only,
                         sort=sort, return_labels=return_labels, normalize=normalize, path=path,
                         return_conditional=return_conditional)
        self.padding = padding
        self.keep_original = keep_original

        if self.mode == DatasetModes.PATH:
            self.path = path
        elif self.mode == DatasetModes.ALL:
            self.path = PATH_ALL
        elif self.mode == DatasetModes.TRAIN:
            self.path = PATH_TRAIN
        elif self.mode == DatasetModes.TEST:
            self.path = PATH_TEST
        else:
            raise ValueError(f"Unknown mode '{self.mode}'")

        # Read data
        log.info(f"Reading trajectories from '{self.path}'.")
        df = pd.read_csv(self.path, dtype={TID: str, UID: 'int32', HOUR: 'int32', DAY: 'int32', CAT: 'int32'})

        # Verify correct format
        num_cat = df[CAT].nunique() if CAT in df else 0
        assert HOUR not in df or df[HOUR].nunique() <= 24, "More than 24 values for Hour of Day"
        assert DAY not in df or df[DAY].nunique() <= 7, "More than 7 values for Day of Week"

        if self.keep_original:
            for tid, t in df.groupby(TID):
                self._originals[tid] = t

        self.features = [SPATIAL_FEATURE, ]
        if HOUR in df and not latlon_only:
            self.features.append(HOUR)
        if DAY in df and not latlon_only:
            self.features.append(DAY)
        if CAT in df and not latlon_only:
            self.features.append(CAT)

        # Determine global reference point and scale factor
        if reference_point is None:
            # Calculate independent reference point per column as max + min / 2
            self.reference_point = (df[spatial_columns].max().values + df[spatial_columns].min().values) / 2    # type: ignore
            # Convert to dict
            self.reference_point = {spatial_columns[i]: self.reference_point[i] for i in range(len(spatial_columns))}
        if scale_factor is None:
            # Individual per axis scale factor as max(abs(df[spatial_columns] - reference_point)))
            self.scale_factor = (df[spatial_columns] - self.reference_point).abs().max().values    # type: ignore
            self.scale_factor = {spatial_columns[i]: self.scale_factor[i] for i in range(len(spatial_columns))}
        self.reference_point = cast(Union[Sequence[float], Dict[str, float], Dict[int, float]],
                                    self.reference_point)  # type: ignore
        self.scale_factor = cast(Union[Sequence[float], Dict[str, float], Dict[int, float]],
                                 self.scale_factor)  # type: ignore

        for tid, df in df.groupby(TID):
            self.tids.append(tid)
            self._uids[tid] = df[UID].iloc[0]
            lat_lon = df[spatial_columns]
            if normalize:
                lat_lon, _, _ = normalize_points(lat_lon, ref=self.reference_point, scale_factor=self.scale_factor,
                                                 columns=spatial_columns)
            lat_lon = torch.from_numpy(lat_lon.to_numpy())
            if latlon_only:
                encoding = lat_lon
            else:
                encoding = [lat_lon, ]
                if HOUR in df:
                    encoding.append(torch.from_numpy(np.eye(24)[df[HOUR].to_numpy()]))
                if DAY in df:
                    encoding.append(torch.from_numpy(np.eye(7)[df[DAY].to_numpy()]))
                if CAT in df:
                    encoding.append(torch.from_numpy(np.eye(num_cat)[df[CAT].to_numpy()]))
            self._encodings[tid] = encoding
            if self.return_conditional:
                cond_info = compute_conditional_info(
                    traj=df, bbox=BBOX_FS, lat=LAT, lon=LON, hour=HOUR, dow=DAY, categorical_timestamp=True,
                    as_dict=True,
                )
                if self.normalize:
                    cond_info = normalise_conditional_info(cond_info, MEAN_COND_FS, STD_COND_FS)  # type: ignore
                self._conditional_info[tid] = cond_info

        if sort:
            self.tids = sorted(self.tids, key=lambda x: len(self._encodings[x]))

        if padding:
            self.lengths = {tid: len(self._encodings[tid]) for tid in self.tids}
            encodings = [self._encodings[tid] for tid in self.tids]
            encodings = pad_sequence(encodings, batch_first=True, padding_value=0.0)
            self._encodings = {tid: encodings[i] for i, tid in enumerate(self.tids)}

        assert len(self.tids) == len(self._encodings), f"{len(self.tids)} != {len(self._encodings)}"

    def __getitem__(self, index: int) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
        tid = self.tids[index]
        output = [self._encodings[tid], ]
        if self.padding:
            output.append(self.lengths[tid])
        if self.return_labels:
            output.append(tid)
        elif self.return_conditional:
            conditional_info = self.get_conditional_info(tid)
            output.append(conditional_info)
        return output if len(output) > 1 else output[0]

    def get_encoding(self, tid: str) -> torch.Tensor:
        self.check_tid(tid=tid)
        return self._encodings[tid]

    def get_original(self, tid: str) -> pd.DataFrame:
        self.check_tid(tid=tid)
        if not self.keep_original:
            raise ValueError("Original data not kept.")
        return self._originals[tid]

    def get_conditional_info(self, tid: str) -> torch.Tensor:
        self.check_tid(tid=tid)
        return self._conditional_info[tid]


if __name__ == "__main__":
    # Create object
    ds = FSNYCDataset()
    assert len(ds) == 3079, f"Dataset contains {len(ds)} trajectories but expected 3,079."
    print("Successfully created dataset object")
