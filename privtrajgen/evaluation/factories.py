#!/usr/bin/env python3
"""Create Model and abstract the training process"""
import argparse
import logging
import multiprocessing
import os
import re
from dataclasses import asdict
from pprint import pprint
from timeit import default_timer as timer
from typing import Optional, cast, Any, Callable

import numpy as np
import pandas as pd
import torch
from pydantic import validate_call
from torch import nn
from torch.utils.data import DataLoader

from privtrajgen.config.enums import Models
from privtrajgen.datasets import get_difftraj_dataset
from privtrajgen.datasets.difftraj_dataset import get_conditional_dataset, difftraj_paths
from privtrajgen.models import model_factories, UNetGAN, UNetVAE
from privtrajgen.models.base_gan import BaseGAN, GEN_PLACEHOLDER, DIS_PLACEHOLDER
from privtrajgen.models.base_model import BaseModel
from privtrajgen.models.diffTraj import train_diffTraj, generate_diffTraj
from privtrajgen.models.save_and_load import prepare_parameter_file, load_model, get_last_saved_epoch
from privtrajgen.utils.nn_utils import compute_steps_and_epochs, get_device, get_number_of_gpus
from .config import BaseConfig, UNetGANConfig, UNetVAEConfig, DiffTrajConfig, ALL_CONFIGS, PrivTraceConfig, Config
from .split_dataset import get_train_set, get_test_set, get_train_set_len
from ..config import LOG_DIR, BASE_DIR, ConditionalInfoSource, RESULT_DIR
from ..models.naming import generate_model_name
from ..utils import configure_root_loger, time_to_str

log = logging.getLogger()


def logc(header: str, *args: Any, column_width: int = 40) -> None:
    log.info(f"{header:<{column_width}}" + "".join([str(arg) for arg in args]))


def get_train_dataloader(config: BaseConfig) -> DataLoader:
    """Load a dataloader from the config"""
    # Load data
    dataset = get_difftraj_dataset(config.dataset, conditional=config.conditional)

    # K-Fold Cross Validation
    dataset = get_train_set(dataset_name=config.dataset, split=config.run_id - 1, data=dataset, n_splits=config.k_fold)

    # Create DataLoader (no collate_fn needed)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=8,
                            drop_last=True, pin_memory=True)
    return dataloader


def get_test_dataset(config: BaseConfig | PrivTraceConfig) -> np.ndarray:
    """Load the real dataset for validation."""
    dataset = np.load(difftraj_paths[config.dataset])

    # K-Fold Cross Validation
    dataset = get_test_set(dataset_name=config.dataset, split=config.run_id - 1, data=dataset, n_splits=config.k_fold)

    return dataset


def get_test_context(config: BaseConfig) -> Optional[torch.Tensor]:
    """Load the conditional information for validation."""
    if config.conditional:
        if config.cond.source == ConditionalInfoSource.DIFFTRAJ:
            context = get_conditional_dataset(config.dataset)
        elif config.cond.source == ConditionalInfoSource.SAMPLE:
            # Use entire samples as conditional information
            context = np.load(difftraj_paths[config.dataset])
            # Convert to tensor
            context = torch.tensor(context, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown conditional information source {config.cond.source}")
        # K-Fold Cross Validation
        context = get_test_set(dataset_name=config.dataset, split=config.run_id - 1, data=context,
                               n_splits=config.k_fold)
        return context
    else:
        return None


def get_model(config: BaseConfig) -> BaseModel:
    """Create a model from the config"""
    if config.model_type not in model_factories:
        raise ValueError(f"Unknown model {config.model_type}")
    return model_factories[config.model_type](config)


def load_parameters(config: BaseConfig, model: nn.Module) -> nn.Module:
    """Load the model parameters"""
    device = get_device(config.gpu)
    # Move model to device just to be sure
    model.to(device)

    # Determine epoch
    if config.validation.epoch:
        epoch = config.validation.epoch
    else:
        # Use the last epoch
        train_set_size = get_train_set_len(dataset=config.dataset, n_splits=config.k_fold)
        # Compute steps per epoch: Number of batches, and last batch is dropped
        steps_per_epoch = train_set_size // config.batch_size
        _, epoch = compute_steps_and_epochs(config.steps, config.epochs, steps_per_epoch=steps_per_epoch)

    # Print a warning if the epoch is not the last one
    last_epoch = get_last_saved_epoch(config)
    if last_epoch != epoch:
        log.warning(f"Loading epoch {epoch} instead of the last saved epoch {last_epoch}.")

    # Load parameters
    if isinstance(model, BaseGAN):
        # Determine parameter path
        param_file = prepare_parameter_file(modelname=config.name, use_model_placeholder=True)
        # Generator
        load_model(
            model=model.gen,
            param_path=param_file,
            epoch=epoch,
            model_placeholder=GEN_PLACEHOLDER,
            device=device,
            is_ema=config.validation.ema,
            verbose=True,
        )
        # Discriminator
        load_model(
            model=model.dis,
            param_path=param_file,
            epoch=epoch,
            model_placeholder=DIS_PLACEHOLDER,
            device=device,
            is_ema=config.validation.ema,
            verbose=True,
        )
    else:
        # Determine parameter path
        param_file = prepare_parameter_file(modelname=config.name)
        load_model(
            model=model,
            param_path=param_file,
            epoch=epoch,
            device=device,
            is_ema=config.validation.ema,
            verbose=True,
        )

    return model


def train_model(config: BaseConfig, model: nn.Module, dataloader: DataLoader) -> None:
    """Train the model"""
    logc("Start Training of Model:", config.name)

    # Move model to device
    device = get_device(config.gpu)
    model.to(device)
    logc("Device:", device)

    if isinstance(model, BaseGAN) and config.model_type == Models.UNetGAN:
        config = cast(UNetGANConfig, config)
        # Train
        model.training_loop(
            dataloader=dataloader,
            config=config,
        )
    elif isinstance(model, UNetVAE):
        config = cast(UNetVAEConfig, config)
        # Train
        model.training_loop(dataloader=dataloader, config=config)
    elif config.model_type == Models.DiffTraj:
        config = cast(DiffTrajConfig, config)
        train_diffTraj(config, model, dataloader)
    else:
        raise ValueError(f"Unknown model {config.model_type}")


def generate_trajectories(config: BaseConfig, model: torch.nn.Module, num: int,
                          head: Optional[torch.Tensor]) -> np.ndarray:
    """Generate trajectories from the model.

    :param config: The configuration of the model.
    :param model: The model to generate the trajectories from.
    :param num: The number of trajectories to generate.
    :param head: The conditional information to use for the generation or None.
    :return: The generated trajectories.
    """
    if head is not None and len(head) != num:
        raise ValueError(f"Expected {num} conditional information tensors, got {len(head)}.")

    # Move model (and head) to device
    device = get_device(config.gpu)
    model.to(device)
    head = head.to(device, dtype=torch.float32) if head is not None else None

    # Set model to eval mode
    model.eval()

    # Generate the trajectories
    if config.model_type == Models.UNetGAN:
        model = cast(UNetGAN, model)
        z = model.gen.get_noise(batch_size=num)
        syn_data = model.forward(z, head).detach().cpu().numpy()
    elif config.model_type == Models.UNetVAE:
        model = cast(UNetVAE, model)
        syn_data = model.sample(num, context=head).detach().cpu().numpy()
    elif config.model_type == Models.DiffTraj:
        assert isinstance(config, DiffTrajConfig)
        syn_data = generate_diffTraj(model=model, config=config, conditional_attr=head, n_total=num,
                                     batch_size=min(num, 5096))
    else:
        raise ValueError(f"Unknown model {config.model_type}")

    syn_data = np.array(syn_data)
    return syn_data


def get_argparser() -> argparse.ArgumentParser:
    """Create the argument parser for the evaluation scripts."""
    parser = argparse.ArgumentParser(description='Run the evaluation.')
    parser.add_argument('-g', '--gpu', type=int, choices=range(get_number_of_gpus()),
                        help='GPU id to use for the evaluation, default is 0.', default=0)
    # Choose the configuration ID
    parser.add_argument('-c', '--config', type=str, choices=ALL_CONFIGS.keys(),
                        help='The configuration to run.')
    # Choose the run ID
    parser.add_argument('-r', '--run', type=int, help='The run ID to start from.', default=None)
    # dry-run: Only print the configuration
    parser.add_argument('--dry-run', action='store_true', help='Only print the configuration.')
    return parser


def save_runtime(config: BaseConfig, runtime: float) -> None:
    """Save the runtime of the training.

    :param config: Configuration of the model
    :param runtime: Runtime in seconds
    """
    run_name = generate_model_name(config)
    # Remove the run component from the name - store all runs into the same file as a table
    run_name = re.sub(r"_run\d{2}", "", run_name)

    # Append the runtime to the file (or create if not exists)
    res_dir = os.path.join(BASE_DIR, 'results/')
    os.makedirs(res_dir, exist_ok=True)  # Ensure the directory exists
    file = os.path.join(res_dir, f'{run_name}_runtime.csv')

    # Columns: run_id, runtime
    df = pd.DataFrame(data={'run_id': [config.run_id], 'runtime': [runtime]})

    if os.path.exists(file):
        # Read the existing file to ensure column order
        existing_df = pd.read_csv(file)
        # Reorder the new DataFrame to match the existing column order
        df = df[existing_df.columns]

    # Write to the file
    df.to_csv(
        file,
        index=False,
        mode='a',  # Append to the file
        header=not os.path.exists(file)  # Only write the header if the file does not exist
    )


def load_runtimes_from_file(file: str) -> pd.DataFrame:
    if os.path.exists(file):
        return pd.read_csv(file)
    else:
        return pd.DataFrame()

def load_runtimes_by_name(run_name: str) -> pd.DataFrame:
    """Load the runtimes of the training by run name.

    :param run_name: Name of the run
    :return: DataFrame with the runtimes
    """
    # Remove the run component from the name - store all runs into the same file as a table
    run_name = re.sub(r"_run\d{2}", "", run_name)

    file = os.path.join(RESULT_DIR, f'{run_name}_runtime.csv')
    return load_runtimes_from_file(file)

def load_runtimes(config: Config) -> pd.DataFrame:
    """Load the runtimes of the training.

    :param config: Configuration of the model
    :return: DataFrame with the runtimes
    """
    if isinstance(config, BaseConfig):
        run_name = generate_model_name(config)
    elif isinstance(config, PrivTraceConfig):
        run_name = config.name
    else:
        raise ValueError(f"Unknown config type {type(config)}")
    return load_runtimes_by_name(run_name)

@validate_call(config={'arbitrary_types_allowed': True})
def execute(subprocess: Callable[[BaseConfig], None], operation: str) -> None:
    # Configure the logger
    configure_root_loger(logging.INFO, os.path.join(LOG_DIR, f'{operation}.log'))
    # Parse the arguments
    args = get_argparser().parse_args()
    configs = ALL_CONFIGS[args.config]

    # Make sure configs is a list
    if not isinstance(configs, list):
        configs = [configs]

    # Start subprocesses for each model
    processes = []
    for config in configs:
        # Set the GPU
        config.gpu = args.gpu
        if args.dry_run:
            pprint(asdict(config))
            continue
        # Override the run ID - allows to resume from a specific run
        if args.run is not None:
            start_id = args.run
        else:
            start_id = config.run_id
        log.info(f"Starting run ID {start_id}.")
        for run in range(start_id, config.num_runs + start_id):
            # Update the run id & seed
            config.run_id = run
            config.seed = config.seed + run
            # Start timer
            start_time = timer()
            if len(configs) == 1 and config.num_runs == 1:
                # We don't need to create a subprocess - helps with HPC issues
                subprocess(config)
            else:
                # Start the subprocess
                p = multiprocessing.Process(target=subprocess, args=(config,))
                p.start()
                processes.append(p)
                # Wait for this process to finish before starting the next one b/c of GPU memory
                p.join()
            # Log the time
            runtime = timer() - start_time  # in seconds
            log.info(f"Process terminated in {time_to_str(runtime)}")
            # Write the training duration into a file
            if operation == "train":
                save_runtime(config, runtime)
