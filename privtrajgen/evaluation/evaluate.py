#!/usr/bin/env python3
"""Load a trained model and perform utility measurements."""
import logging
import os
from dataclasses import asdict
from pprint import pprint
from timeit import default_timer as timer
from typing import Union, Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from privtrajgen.config import BASE_DIR, EVALUATION_TRAJECTORY_NUM, LOG_DIR, ConditionalInfoSource
from privtrajgen.datasets import dataset_bounding_boxes, denormalise_by_dataset
from privtrajgen.metrics import compute_all_metrics, save_metrics
from privtrajgen.models.naming import generate_model_name
from privtrajgen.utils import add_filehandler, count_outside_bbox, set_random_seed, time_to_str
from privtrajgen.utils.nn_utils import get_device
from .config import BaseConfig
from .factories import get_model, get_test_context, load_parameters, get_test_dataset, generate_trajectories, \
    execute

log = logging.getLogger()


def logc(header: str, *args: Any, column_width: int = 40, level: int = logging.INFO) -> None:
    txt = f"{header:<{column_width}}" + "".join([str(arg) for arg in args])
    log.log(level, txt)


def evaluate_model(config: BaseConfig, model: torch.nn.Module, run: int = 1) -> None:
    """Evaluate the model on the given dataset."""
    traj_len = config.data.traj_length

    # Load real trajectories as baseline for the metrics
    real_dataset = get_test_dataset(config)
    # Retrieve conditional dataset (Will be None if not conditional)
    head: Union[None, Tensor] = get_test_context(config)

    # Check that if the dataset is conditional and the source is SAMPLE, the head should be equal to the real dataset
    if head is not None and config.cond.source == ConditionalInfoSource.SAMPLE:
        assert np.allclose(real_dataset, head), (
            f"Using SAMPLE mode for conditional information, but head is not equal to the real dataset."
        )

    # Check that both datasets have the same length (Unless head is None)
    if head is not None:
        assert len(real_dataset) == len(head), (
            f"Real and conditional datasets have different lengths: {len(real_dataset)} != {len(head)}"
        )

    # Note that both these datasets are ordered in the same way, i.e., real_dataset[i] corresponds to head[i]
    idx = np.random.choice(len(real_dataset), size=EVALUATION_TRAJECTORY_NUM, replace=False)
    real_dataset = real_dataset[idx]
    assert real_dataset.shape == (EVALUATION_TRAJECTORY_NUM, 2, traj_len), (
        f"Expected shape {(EVALUATION_TRAJECTORY_NUM, 2, traj_len)}, got {real_dataset.shape}."
    )
    # Head might be None -> Only select if it is not None
    head = head[idx] if head is not None else None

    # Set model into eval mode b/c certain features won't work correctly in training mode, in particular,
    # the noise schedule for the conditional embedding is not DP!
    model.eval()

    # Generate trajectories
    syn_dataset = generate_trajectories(config, model, num=EVALUATION_TRAJECTORY_NUM, head=head)
    assert syn_dataset.shape == real_dataset.shape, f"Expected shape {real_dataset.shape}, got {syn_dataset.shape}."

    # Reshape both datasets from (B, 2, L) to (B, L, 2)
    real_dataset = real_dataset.transpose(0, 2, 1)
    syn_dataset = syn_dataset.transpose(0, 2, 1)

    # Assert that both datasets have exactly the same shape
    assert real_dataset.shape == syn_dataset.shape, (f"Real and synthetic datasets have different shapes: "
                                                     f"{real_dataset.shape} != {syn_dataset.shape}.")

    # Count how many values are outside the expected range [-1, 1] (should not happen vor VAE & GAN, but unavoidable
    # for the Diffusion model)
    real_outside = np.sum(np.abs(real_dataset) > 1)
    syn_outside = np.sum(np.abs(syn_dataset) > 1)
    # The real dataset should not have values outside [-1, 1]. If it does, the data is not normalised correctly, and
    # we should not proceed but raise an assertion error.
    assert real_outside == 0, (f"Real dataset has values outside [-1, 1]: "
                               f"{real_outside}/{real_dataset.shape[0] * real_dataset.shape[1]}")

    # Also, only in case of diffusion models, the synthetic dataset might have values outside [-1, 1].
    if config.is_diffusion and syn_outside > 0:
        logc("Synthetic dataset has values outside [-1, 1]: ",
             f"{syn_outside}/{syn_dataset.shape[0] * syn_dataset.shape[1]}", level=logging.INFO)
        logc("This is valid for diffusion models, but not for VAEs, GANs, or any other model "
             "that use a tanh activation as final layer.", level=logging.INFO)
        # Create a plot and save to img/eval_bbox_error.png for inspection
        points = syn_dataset.reshape(-1, 2)
        plt.scatter(points[:, 0], points[:, 1])
        directory = os.path.join(BASE_DIR, "img/")
        os.makedirs(directory, exist_ok=True)
        plt.savefig(os.path.join(directory, "eval_bbox_error.png"))
        # This is not an error, but we should still fix it to not have unexpected results during evaluation
        # Clip the values to [-1, 1]
        syn_dataset = np.clip(syn_dataset, -1, 1)
    else:
        assert syn_outside == 0, (f"Synthetic dataset has values outside [-1, 1]: "
                                  f"{syn_outside}/{syn_dataset.shape[0] * syn_dataset.shape[1]} = "
                                  f"{syn_outside / (syn_dataset.shape[0] * syn_dataset.shape[1]) * 100:.2f}%")

    # Denormalise the datasets into geographical coordinates
    real_dataset = denormalise_by_dataset(real_dataset, config.dataset)
    syn_dataset = denormalise_by_dataset(syn_dataset, config.dataset)

    # Retrieve the dataset-specific bounding box
    bounding_box = dataset_bounding_boxes[config.dataset]

    # Check how many values are outside the bounding box
    # Assume the last dimension is in (lon, lat) order, and bounding box is dict (lon|lat)_(min|max) -> value
    # As we fixed the data above, we should not have any values outside the bounding box
    real_outside = count_outside_bbox(real_dataset, bounding_box)
    syn_outside = count_outside_bbox(syn_dataset, bounding_box)
    assert real_outside == 0, f"Real dataset has {real_outside} values outside the denormalised bounding box."
    assert syn_outside == 0, f"Synthetic dataset has {syn_outside} values outside the denormalised bounding box."

    # Compute the metrics
    results = compute_all_metrics(
        trajectory_dataset1=real_dataset,
        trajectory_dataset2=syn_dataset,
        lon_first=True,
        geographical_coordinates=True,
        are_matched=config.conditional,  # If we use conditional information, the matching is clear
        bbox=bounding_box,
        random_seed=config.seed,
    )

    # Save metrics to file
    filename = f"{config.name}_{run:02d}"
    save_metrics(
        results=results,
        filename=filename,
        metadata=asdict(config),
        per_traj_metrics=True,
    )


def validate_model_subprocess(config: BaseConfig) -> None:
    # Set the matplotlib backend to Agg to fix multiprocessing error on HPC
    matplotlib.use('Agg')

    # Generate model name / Run ID might have changed
    config.name = generate_model_name(config)

    # Add logfile based on model name
    add_filehandler(file=os.path.join(LOG_DIR, f"{config.name}_validate.log"))
    # Set logging level to DEBUG
    logging.getLogger().setLevel(logging.DEBUG)

    # Log the configuration
    log.info(f"Configuration: {config}")

    # Pretty print the config
    pprint(asdict(config))

    # Set the random seeds
    set_random_seed(config.seed)

    # Retrieve the model
    model = get_model(config)

    # Move the model to the device
    model.to(get_device(config.gpu))

    # Load the model parameters
    model = load_parameters(config, model)

    # Evaluate the model
    for run in range(1, config.validation.runs + 1):
        start = timer()
        evaluate_model(config, model, run=run)
        log.info(f"Evaluation terminated in {time_to_str(timer() - start)}")


if __name__ == '__main__':
    execute(subprocess=validate_model_subprocess, operation="evaluate")
