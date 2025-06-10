#!/usr/bin/env python3
""" """
from typing import List, Union, Optional, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

from privtrajgen.evaluation.config import DiffTrajConfig
from .utils import get_difftraj_model
from .utils import p_xt


def load_diffTraj(
        config: DiffTrajConfig,
        param_file: str,
        epoch: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None
) -> nn.Module:
    """
    Load a DiffTraj model from a parameter file.
    :param config: The configuration of the model
    :param param_file: The file containing the model parameters
    :param epoch: The epoch to load (only required if file contains placeholder)
    :param device: The device to use (e.g., 'cuda:0' or 'cpu')
    :return: The loaded model
    """
    # Set up the device (read from config if not provided)
    if device is None:
        device = torch.device(f"cuda:{config.gpu}" if torch.cuda.is_available() else "cpu")

    # Create Model
    model = get_difftraj_model(config).to(device=device)

    # Load parameters
    if epoch is not None:
        param_file = param_file.format(EPOCH=epoch)

    model.load_state_dict(torch.load(param_file, map_location=device, weights_only=True))

    return model.to(device=device)


def generate_diffTraj(
        model: nn.Module,
        config: DiffTrajConfig,
        conditional_attr: Optional[Union[np.ndarray, torch.Tensor]] = None,
        n_total: int = 1000,
        batch_size: int = 5096,
) -> List[np.ndarray]:
    """
    Generate trajectories using the DiffTraj model.
    :param model: The DiffTraj model
    :param config: The configuration of the model
    :param conditional_attr: The conditional attributes to use (if the model is conditional)
    :param n_total: The total number of trajectories to generate (overwritten if conditional_attr is given)
    :param batch_size: The batch size to use
    :return: A list of generated trajectories with shape (n_total, traj_length, 2)
    """
    device = next(model.parameters()).device
    assert config.conditional == (conditional_attr is not None), "Conditional model requires conditional attributes."
    # Create DataLoader from attr, if given (otherwise, use default)
    if conditional_attr is not None:
        # Randomly select n_total samples from the conditional attributes
        n_total = min(n_total, len(conditional_attr))
        if n_total < len(conditional_attr):
            selected = np.random.choice(len(conditional_attr), n_total, replace=False)
            conditional_attr = conditional_attr[selected]
        if not isinstance(conditional_attr, torch.Tensor):
            conditional_attr = torch.tensor(conditional_attr).to(dtype=torch.float32)
        dataset = conditional_attr
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    else:
        # Use an empty dataset
        dataset = [None] * n_total
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)

    # Prediction Parameters -> based on diffTraj/process/Traj_Generation.ipynb
    n_steps = config.diffusion.num_diffusion_timesteps
    beta = torch.linspace(config.diffusion.beta_start, config.diffusion.beta_end, n_steps).to(device=device)
    eta = config.diffusion.eta
    timesteps = config.diffusion.num_generation_timesteps
    skip = n_steps // timesteps  # For speed up, we skip some steps during generation
    seq = range(0, n_steps, skip)
    seq_next = [-1] + list(seq[:-1])

    generated_trajs = []
    for attributes in tqdm(dataloader, total=len(dataloader), desc="Batches", leave=False):
        model.eval()

        # Get batch size (depends on conditional information)
        batch_size = len(attributes)

        # Move conditional data to device
        attributes = attributes.to(device=device, dtype=torch.float32) if config.conditional else None

        # Start with random noise
        x = torch.randn(batch_size, 2, config.data.traj_length).to(device=device, dtype=torch.float32)

        ims = []
        for i, j in tqdm(zip(reversed(seq), reversed(seq_next)), total=len(seq), leave=False):
            t = (torch.ones(batch_size) * i).to(x.device)
            next_t = (torch.ones(batch_size) * j).to(x.device)
            with torch.no_grad():
                pred_noise = model(x, t, attributes)
                x = p_xt(x, pred_noise, t, next_t, beta, eta)
                if i % 10 == 0:
                    ims.append(x.cpu().squeeze(0))

        trajs = ims[-1].cpu().numpy()
        generated_trajs.extend(trajs)
    return generated_trajs


def plot_trajectories(trajectories: List[np.ndarray], title: Optional[str] = None, alpha: float = 0.1,
                      connected: bool = True,
                      ax: Optional[plt.Axes] = None) -> Tuple[Optional[plt.Figure], plt.Axes]:
    """
    Plot a list of trajectories.
    :param trajectories: List of trajectories as numpy arrays either of shape (n, 2) or (2, n)
    :param title: The title of the plot
    :param alpha: The alpha value of the plot
    :param connected: Whether to connect the points
    :param ax: The axis to plot on (if None, a new figure is created)
    :return: (fig, ax)
    """
    # Check if it is necessary to transpose the trajectories
    if trajectories[0].shape[-1] != 2:
        trajectories = [traj.T for traj in trajectories]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        show = True
    else:
        fig, show = None, False
    for i in range(len(trajectories)):
        traj = trajectories[i]
        if connected:
            ax.plot(traj[:, 0], traj[:, 1], color='blue', alpha=alpha)
        else:
            ax.scatter(traj[:, 0], traj[:, 1], color='blue', alpha=alpha, s=1)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    if title is not None:
        ax.set_title(title)
    if show:
        plt.tight_layout()
    return fig, ax
