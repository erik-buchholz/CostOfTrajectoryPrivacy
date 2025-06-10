#!/usr/bin/env python3
"""PyTorch Wrapper for GeoLife Dataset by Microsoft.
URL: [Microsoft Research](https://www.microsoft.com/en-us/download/details.aspx?id=52367)
"""
import argparse
import logging
import multiprocessing as mp
from collections import OrderedDict
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple, Union, Sequence, Dict, cast, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import privtrajgen.datasets.preprocessing as pp
from privtrajgen import config
from privtrajgen.config import DatasetModes, Datasets
from privtrajgen.config.constants import *
from privtrajgen.utils import normalize_points
from privtrajgen.utils.helpers import find_bbox, get_ref_point, get_scaling_factor
from .base_dataset import TrajectoryDataset
from .conditional_info import compute_conditional_info, normalise_conditional_info
from .constants import NUM_PROCESSED_TRAJS_GEOLIFE, MAX_LEN_GEOLIFE, DEFAULT_REF_POINT_GEOLIFE, \
    DEFAULT_SCALE_FACTOR_GEOLIFE
from .utils import DictProperty

GEOLIFE_DIR = config.BASE_DIR + 'data/geolife/Data/'
NUM_FILES_GEOLIFE = 18670  # Determined via `lsl data/geolife/Data/???/Trajectory/ | grep '.plt' | wc -l`
SAMPLING_RATE_GEOLIFE = 5  # [s]
MAXIMAL_BREAK_INTERVAL_GEOLIFE = 60  # [s]
MIN_LEN_GEOLIFE = 10
# BBOX_QUANTILE = 0.95
GEOLIFE_DATATYPES = OrderedDict({
    "latitude": float,
    "longitude": float,
    "ignore": str,
    "altitude": float,
    "days": float,
    "date": str,
    "time": str,
})
log = logging.getLogger()
# Reference Point and Scale Factor
# Due to dataset size, on-demand computation is very expensive
# 90-Percentile BBOX
GL_REF_POINT_90 = None
GL_SCALE_FACTOR_90 = None
BBOX_GEOLIFE_FIFTH = {
    'lat_min': 39.75,
    'lat_max': 40.03,
    'lon_min': 116.19,
    'lon_max': 116.56
}
MEAN_COND_GEOLIFE = {
    'avg_dist': 22.96916242620554,
    'avg_speed': 3.851351186132182,
    'len': 93.32442737108656,
    'total_dist': 2098.783188510363,
    'total_dur': 498.63866635425313}
STD_COND_GEOLIFE = {'avg_dist': 29.636925841041908,
                    'avg_speed': 4.271180210230596,
                    'len': 67.70581764917506,
                    'total_dist': 3087.0356173426235,
                    'total_dur': 347.30880350842705}

DEFAULT_DATASET_PATH_GEOLIFE = config.BASE_DIR + "data/geolife_FIFTH-RING_5_60_200_TRUNCATE"
DEFAULT_BBOX_GEOLIFE = BBOX_GEOLIFE_FIFTH
DEFAULT_BBOX_NAME_GEOLIFE = "FIFTH-RING"

GL_BBOXES = {
    90: (115.65822549999999, 116.688371, 36.3831297, 40.072966),
    95: (108.4700764500001, 117.967031749, 30.847590175850016, 40.371033),
    'FIFTH-RING': (116.19, 116.56, 39.75, 40.03),  # Fifth ring road of Beijing
    'SIXTH-RING': (116.07, 116.72, 39.68, 40.18),  # Sixth ring road of Beijing
}


class TrajectoryLength(Enum):
    REMOVE = 1
    TRUNCATE = 2
    SPLIT = 3


LENGTH_METHOD = TrajectoryLength.TRUNCATE


def get_filename(bbox_name: str, sampling_rate: int | float, maximal_break_interval: int | float, max_len: int | float,
                 length_method: TrajectoryLength) -> str:
    return f"geolife_{bbox_name}_{sampling_rate}_{maximal_break_interval}_{max_len}_{length_method.name}"


class GeoLifeDataset(TrajectoryDataset):
    """
    Pre-processed GeoLife dataset
    """

    def __init__(self,
                 mode: DatasetModes = DatasetModes.PATH,
                 path: Optional[str] = None,
                 spatial_columns: list = SPATIAL_COLUMNS,
                 latlon_only: bool = True,
                 normalize: bool = True,
                 return_labels: bool = False,
                 sort: bool = False,
                 keep_original: bool = False,
                 as_dataframe: bool = False,
                 reference_point: Union[
                     Sequence[float], Dict[str, float], Dict[int, float]] = DEFAULT_REF_POINT_GEOLIFE,
                 scale_factor: Union[
                     Sequence[float], Dict[str, float], Dict[int, float]] = DEFAULT_SCALE_FACTOR_GEOLIFE,
                 return_conditional: bool = False,
                 ):
        # Validate inputs
        if sort:
            raise NotImplementedError("Sorting is not implemented due to size of GeoLife dataset.")

        # Call superclass / store parameters
        features = [SPATIAL_FEATURE] if latlon_only else [SPATIAL_FEATURE, HOUR, DAY]
        super().__init__(spatial_columns=spatial_columns, name=Datasets.GEOLIFE, max_len=MAX_LEN_GEOLIFE,
                         reference_point=reference_point, scale_factor=scale_factor, mode=mode, features=features,
                         latlon_only=latlon_only, sort=sort, return_labels=return_labels, normalize=normalize,
                         path=path, keep_original=keep_original, return_conditional=return_conditional)
        self.as_dataframe = as_dataframe

        log.warning("Make sure the reference point and scale factor computed during pre-processing are used!")

        if self.mode == DatasetModes.ALL:
            self.path: str = DEFAULT_DATASET_PATH_GEOLIFE
        elif self.mode != DatasetModes.PATH:
            raise NotImplementedError("Only path mode is supported at the moment.")

        log.info(f"Reading trajectories from '{self.path}'.")
        path = Path(self.path)
        for child in path.glob("*.csv"):
            tid = child.name.replace('.csv', '')
            self.tids.append(tid)

        self.tids = sorted(self.tids)

        # Original DataFrames are generated on demand --> replace dict by property
        self._originals = {}
        self.originals = DictProperty(self._originals, self.tids, self._encode)

        if NUM_PROCESSED_TRAJS_GEOLIFE is not None:
            assert len(self.tids) == NUM_PROCESSED_TRAJS_GEOLIFE, (f"Found {len(self.tids)} files, "
                                                                   f"expected {NUM_PROCESSED_TRAJS_GEOLIFE}.")

    def _encode(self, tid: str) -> None:
        # Load from file without parsing dates
        df = pd.read_csv(
            f"{self.path}/{tid}.csv",
            dtype={
                TID: str,
                UID: int,
                LAT: float,
                LON: float,
                ALTITUDE: float,
                TIMESTAMP: str
            }
        )

        # Function to handle two potential date formats
        def try_parse_dates(dates: pd.Series) -> pd.Series:
            # Try first format
            parsed_dates = pd.to_datetime(dates, format="%Y-%m-%d %H:%M:%S.%f", errors='coerce')
            # Where parsing failed, try second format
            parsed_dates = parsed_dates.fillna(pd.to_datetime(dates, format="%Y-%m-%d %H:%M:%S", errors='coerce'))
            return parsed_dates

        # Apply the date parsing function to the timestamp column
        df[TIMESTAMP] = try_parse_dates(df[TIMESTAMP])

        # Check if timestamp parsing was successful
        if df[TIMESTAMP].isnull().any():
            raise ValueError("Timestamp parsing failed for some entries.")
        if self.keep_original:
            self._originals[tid] = df
        self._uids[tid] = df[UID][0]
        if self.as_dataframe:
            self._encodings[tid] = df
        else:
            # Convert to Tensor
            latlon = df[self.columns]
            if self.normalize:
                latlon, _, _ = normalize_points(latlon, ref=self.reference_point, scale_factor=self.scale_factor,
                                                columns=self.columns)
            latlon = torch.from_numpy(latlon.to_numpy())
            if self.latlon_only:
                self._encodings[tid] = latlon
            else:
                # Transform timestamp into one hot encoding for hour of day and day of week
                hour = torch.from_numpy(np.eye(24)[df[TIMESTAMP].dt.hour.to_numpy()])
                day = torch.from_numpy(np.eye(7)[df[TIMESTAMP].dt.dayofweek.to_numpy()])
                self._encodings[tid] = [latlon, hour, day]
        if self.return_conditional:
            # Conditional Information
            cond_info = cast(dict,
                             compute_conditional_info(traj=df, bbox=BBOX_GEOLIFE_FIFTH, lat=LAT, lon=LON,
                                                      as_dict=True, ))
            if self.normalize:
                cond_info = normalise_conditional_info(cond_info, MEAN_COND_GEOLIFE, STD_COND_GEOLIFE)  # type: ignore
            self._conditional_info[tid] = cond_info

    def __getitem__(self, index: int) -> Union[
        Tuple[torch.Tensor, str], Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
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


def _read_geolife_file(root_dir: str, uid: int, tid: str) -> pd.DataFrame:
    """Read the original GeoLife files into dataFrame.
    :param root_dir: Root directory of the GeoLife dataset
    :param uid: User ID
    :param tid: Trajectory ID
    :return: DataFrame containing the trajectory
    """
    filename = f"{root_dir}/{uid:03d}/Trajectory/{tid}.plt"
    dates = [['date', 'time']]
    df = pd.read_csv(
        filename,
        delimiter=',',
        header=None,
        names=list(GEOLIFE_DATATYPES.keys()),
        skiprows=6,
        dtype=GEOLIFE_DATATYPES,
        parse_dates=dates
    )
    df.rename(columns={'date_time': TIMESTAMP, 'latitude': LAT, 'longitude': LON}, inplace=True)
    df.drop(inplace=True, columns=['ignore', 'days'])
    df[UID] = uid
    df[TID] = f"{uid:03d}_{tid}"
    return df


class GeoLifeUnprocessed(Dataset):
    """Unprocessed GeoLife dataset"""

    def __init__(self, path: str = GEOLIFE_DIR):
        self.root = path

        self.uids = []
        self.tids = []

        root_dir = Path(self.root)

        # Check if Path exists and raise error if not
        if not root_dir.exists():
            raise ValueError(f"GeoLife dataset not found at {root_dir}."
                             f"Make sure you downloaded it into {GEOLIFE_DIR} or you specified the correct path.")

        for child in root_dir.iterdir():
            if child.is_dir():
                uid = int(child.name)
                self.uids.append(uid)
                for traj in child.glob("Trajectory/*.plt"):
                    t_name = f"{uid:03d}_{traj.name.replace('.plt', '')}"
                    self.tids.append(t_name)

        self.uids = sorted(self.uids)
        self.tids = sorted(self.tids)

        assert len(self.tids) == NUM_FILES_GEOLIFE, (f"Found {len(self.tids)} files, expected {NUM_FILES_GEOLIFE}."
                                                     f"Downloaded dataset may be incomplete.")

    def __len__(self) -> int:
        return len(self.tids)

    def __getitem__(self, index: int) -> pd.DataFrame:
        tid = self.tids[index]
        uid, tid = tid.split('_')
        uid = int(uid)
        return _read_geolife_file(self.root, uid, tid)


def wrap_resample(t: Any) -> pd.DataFrame:
    return pp.resample_trajectory(*t)


def preprocess_geolife(
        input_dir: str = GEOLIFE_DIR,
        sampling_rate: int = SAMPLING_RATE_GEOLIFE,
        split_interval: float = MAXIMAL_BREAK_INTERVAL_GEOLIFE,
        min_length: int = MIN_LEN_GEOLIFE,
        max_length: int = MAX_LEN_GEOLIFE,
        length_method: TrajectoryLength = LENGTH_METHOD,
        bbox_quantile: Optional[float | None] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        bbox_name: Optional[str] = None,
        overwrite: bool = False,
        test: bool = False
) -> list[pd.DataFrame]:
    """
    Preprocess GeoLife dataset.
    :param input_dir: Directory containing the unzipped GeoLife dataset
    :param sampling_rate: Sampling rate in seconds
    :param split_interval: Maximum break interval in seconds
    :param min_length: Minimum length of any trajectory
    :param max_length: Maximum length of any trajectory
    :param length_method: How to handle trajectories that are too long
    :param bbox_quantile: Bounding Box quantile. OR
    :param bbox: Bounding Box (instead of quantile): (lon_min, lon_max, lat_min, lat_max) OR
    :param bbox_name: Name of the BBOX (for filename)
    :param overwrite: Overwrite existing files
    :param test: Run in test mode (don't save)
    :return:
    """
    # Validate that exactly one of the three bbox values is defined
    bbox_parameters = [bbox_quantile, bbox, bbox_name]
    non_none_parameters = sum([1 for param in bbox_parameters if param is not None])
    if non_none_parameters != 1:
        # raise ValueError("Exactly one of bbox_quantile, bbox, or bbox_name should be provided.")
        log.warning("Exactly one of bbox_quantile, bbox, or bbox_name should be provided.")
        log.warning(f"Using default bbox: {DEFAULT_BBOX_NAME_GEOLIFE}.")
        bbox_name = DEFAULT_BBOX_NAME_GEOLIFE
    if bbox_quantile is not None:
        bbox_name = f"{bbox_quantile * 100:.0f}-quantile"
    elif bbox is not None:
        bbox_name = f"CUSTOM"
    elif bbox_name is None:
        raise ValueError("Exactly one of bbox_quantile, bbox, or bbox_name should be provided.")

    # Make sure output dir does not exist and create it
    output_dir = config.BASE_DIR + f"data/{get_filename(bbox_name, sampling_rate, split_interval, max_length, length_method)}"
    out_dir = Path(output_dir)
    if out_dir.exists() and not overwrite and not test:
        raise ValueError("Output directory exists.")
    else:
        print(f"Writing to {out_dir}.")
        out_dir.mkdir(parents=True, exist_ok=True)

    # Load unprocessed dataset
    gl_all = GeoLifeUnprocessed(path=input_dir)

    # Make sure the full dataset has been loaded and no files are missing
    assert len(gl_all) == NUM_FILES_GEOLIFE
    print(f"Unprocessed GeoLife Trajectories:\t\t{len(gl_all)}")
    geolife_all = pd.concat(gl_all)
    n_unprocessed = len(geolife_all)
    print(f"Number of points in GeoLife Dataset:\t\t{n_unprocessed:,}")

    # Compute bbox
    if bbox is None:
        if bbox_quantile is not None:
            bbox = find_bbox(geolife_all, bbox_quantile)
        elif bbox_name in GL_BBOXES:
            bbox = GL_BBOXES[bbox_name]
        else:
            raise ValueError(f"Unknown BBOX name {bbox_name}.")
    print(f"Bounding Box: {bbox_name} =\t\t\t{bbox}")

    # Drop out of bounds points
    tasks = ((t, *bbox, LAT, LON, False) for t in gl_all)
    with mp.Pool() as pool:
        geolife_in_bbox = pool.starmap(pp.drop_out_of_bounds,
                                       tqdm(tasks, total=len(gl_all), desc="Drop out-of-bounds"))
    n_in_bbox = len(pd.concat(geolife_in_bbox))
    print(f"Number of Points after dropping out of BBOX:\t{n_in_bbox:,} ({n_in_bbox / n_unprocessed * 100:.2f}%)")

    # Down-sampling
    chunksize = 100  # See geolife.ipynb
    tasks = ((t, sampling_rate) for t in geolife_in_bbox)

    # Multiprocessing code with tqdm progress bar
    geolife_resampled = []
    with mp.Pool() as pool:
        with tqdm(total=len(geolife_in_bbox), desc="Resampling") as pbar:
            for t in pool.imap_unordered(wrap_resample, tasks, chunksize=chunksize):
                if len(t) > 0:
                    geolife_resampled.append(t)
                pbar.update()

    geolife_resampled_df = pd.concat(geolife_resampled)
    n_resampled = len(geolife_resampled_df)
    print(
        f"Number of Points after resampling to {sampling_rate}s:\t{n_resampled:,} "
        f"({n_resampled / n_unprocessed * 100:.2f}%)")
    print(
        f"Number of Trajectories after resampling to {sampling_rate}s:\t{len(geolife_resampled):,} "
        f"({len(geolife_resampled) / len(gl_all) * 100:.2f}%)")

    # Split trajectories on breaks
    geolife_split = []
    for i, t in tqdm(enumerate(geolife_resampled), total=len(geolife_resampled), desc="Splitting trajectories"):
        splits = pp.split_and_rename(t, interval=split_interval, max_len=None)
        geolife_split.extend(splits)

    geolife_split_df = pd.concat(geolife_split)
    n_resampled_split = len(geolife_split_df)
    print(
        f"Number of Points after splitting at {split_interval}s:\t{n_resampled_split:,} "
        f"({n_resampled_split / n_unprocessed * 100:.2f}%)")
    print(
        f"Number of Trajectories after splitting at {split_interval}s:\t{len(geolife_split):,} "
        f"({len(geolife_split) / len(gl_all) * 100:.2f}%)")

    # Upper Length
    pb = tqdm(geolife_split, total=len(geolife_split), desc="Truncating trajectories")

    if length_method == TrajectoryLength.REMOVE:
        # Remove trajectories that are longer than 200 points
        geolife_upper_bounded = [t.reset_index(drop=True) for t in pb if len(t) <= MAX_LEN_GEOLIFE]
    elif length_method == TrajectoryLength.TRUNCATE:
        # Truncate trajectories to MAX_LEN points
        geolife_upper_bounded = [t[:MAX_LEN_GEOLIFE].reset_index(drop=True) for t in pb]
    elif length_method == TrajectoryLength.SPLIT:
        # Split trajectories into multiple trajectories of MAX_LEN points or fewer
        geolife_upper_bounded = []
        for t in pb:
            splits = pp.split_and_rename(t, interval=None, max_len=MAX_LEN_GEOLIFE)
            geolife_upper_bounded.extend(splits)
    else:
        raise ValueError(f"Invalid Trajectory Length Method: {length_method}")

    geolife_upper_bounded_df = pd.concat(geolife_upper_bounded)
    n_upper_bounded = len(geolife_upper_bounded_df)
    print(
        f"Number of Points after upper bounding (via {length_method.name}) to {max_length} points:\t\t"
        f"{n_upper_bounded:,} ({n_upper_bounded / n_unprocessed * 100:.2f}%)")
    print(
        f"Number of Trajectories after upper bounding (via {length_method.name}) to {max_length} points:"
        f"\t{len(geolife_upper_bounded):,} ({len(geolife_upper_bounded) / len(gl_all) * 100:.2f}%)")

    # Remove too short trajectories
    geolife_final = [t for t in geolife_upper_bounded if len(t) >= min_length]

    geolife_final_df = pd.concat(geolife_final)
    n_final = len(geolife_final_df)
    print(f"Final Number of Points:\t\t{n_final:,} ({n_final / n_unprocessed * 100:.2f}%)")
    print(
        f"Final Number of Trajectories:\t{len(geolife_final):,} "
        f"({len(geolife_final) / len(gl_all) * 100:.2f}%)")

    # Sort by uid first and tid second
    geolife_final.sort(key=lambda x: (x[UID].iloc[0], x[TID].iloc[0]))

    if not test:
        # Store the dataframe
        geolife_final_df.to_csv(output_dir + '.csv', index=False)
        # Store into separate files for each trajectory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        for i, t in tqdm(enumerate(geolife_final), total=len(geolife_final), desc="Storing Trajectories"):
            tid = f"{t[TID].iloc[0]}"
            t.to_csv(output_dir + f"/{tid}.csv", index=False)

        print(f"Saved to {output_dir}.")

    print(f"Done. Generated {len(geolife_final):,} GeoLife trajectories.")

    # Determine reference point
    ref_point = get_ref_point(geolife_final_df[SPATIAL_COLUMNS])
    print(f"Reference Point:\t{ref_point}")

    # Determine scaling factor
    scaling_factor = get_scaling_factor(geolife_final_df[SPATIAL_COLUMNS], ref_point)
    print(f"Scaling Factor:\t\t{scaling_factor}")

    return geolife_final


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-d', '--directory', type=str, default=PROCESSED_DIR)
    parser.add_argument('--split_interval', type=float, default=MAXIMAL_BREAK_INTERVAL_GEOLIFE)
    parser.add_argument('--sampling_rate', type=int, default=SAMPLING_RATE_GEOLIFE)
    parser.add_argument('--length_method', type=TrajectoryLength, choices=list(TrajectoryLength),
                        default=LENGTH_METHOD)
    parser.add_argument('--max_len', type=int, default=MAX_LEN_GEOLIFE)
    parser.add_argument('--min_len', type=int, default=MIN_LEN_GEOLIFE)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--test', action='store_true')
    # Add argument group for special BBOXes
    bbox_group = parser.add_mutually_exclusive_group()
    bbox_group.add_argument('-b', '--bbox_quantile', type=float, default=None,
                            help="Bounding Box quantile. All points outside the quantile are dropped.")
    # Add argument for fifth ring
    bbox_group.add_argument('--fifth', action='store_true',
                            help="Use the fifth ring of Beijing as BBOX.")
    # Add argument for sixth ring
    bbox_group.add_argument('--sixth', action='store_true',
                            help="Use the sixth ring of Beijing as BBOX.")
    opt = parser.parse_args()
    if opt.fifth:
        bbox_name: Optional[str] = 'FIFTH-RING'
    elif opt.sixth:
        bbox_name = 'SIXTH-RING'
    else:
        bbox_name = None
    print("Arguments: ", opt)

    preprocess_geolife(
        split_interval=opt.split_interval,
        sampling_rate=opt.sampling_rate,
        min_length=opt.min_len,
        max_length=opt.max_len,
        length_method=opt.length_method,
        bbox_quantile=opt.bbox_quantile,
        bbox_name=bbox_name,
        overwrite=opt.overwrite,
        test=opt.test
    )
