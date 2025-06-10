#!/usr/bin/env python3
"""Training Routine for the DiffTraj Model with DP-SGD."""
import logging
import os
import pprint
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime
from types import SimpleNamespace
from typing import Optional, Union, Any, cast, Tuple

import math
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from opacus.utils.batch_memory_manager import BatchMemoryManager
from pydantic import validate_call
from simple_parsing import parse
from torch import nn
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from privtrajgen import config as base_config
from privtrajgen.config import NORMALISED_BBOX, Optimizer
from privtrajgen.datasets import get_difftraj_dataset
from privtrajgen.evaluation.config.difftraj_config import DiffTrajConfig
from privtrajgen.metrics import sliced_wasserstein_distance, JS_from_points
from privtrajgen.models.cond_emb import split_batch
from privtrajgen.models.diffTraj.utils.Traj_UNet import get_difftraj_model
from privtrajgen.models.dp_wrapper import DP_Wrapper
from privtrajgen.models.ema import EMAHelper
from privtrajgen.models.naming import generate_model_name
from privtrajgen.utils.helpers import dict_to_namespace, time_to_str
from privtrajgen.utils.logger import configure_root_loger
from privtrajgen.utils.nn_utils import compute_steps_and_epochs, get_tensorboard_writer, filter_opacus_warnings, \
    is_validation_step, get_device
from .utils.utils import q_xt_x0
from .validate_difftraj import generate_diffTraj, plot_trajectories

# Set the logging level for 'simple-parsing' to WARNING
logging.getLogger('simple_parsing').setLevel(logging.WARNING)
log = logging.getLogger(__name__)


def logc(header: str, *args: Any, column_width: int = 40) -> None:
    log.info(f"{header:<{column_width}}" + "".join([str(arg) for arg in args]))


# Maximum physical batch size for BatchMemoryManager
# Depends on model --> Only valid for DiffTraj
# MAX_PHYSICAL_BATCH_SIZE = 512  # Server Yennefer (empirical value)
MAX_PHYSICAL_BATCH_SIZE = 800  # HPC (empirical value)


@validate_call(config={"arbitrary_types_allowed": True})
def train_diffTraj(
        config: Union[dict, SimpleNamespace, DiffTrajConfig],
        module: Optional[nn.Module] = None,
        dataloader: Optional[DataLoader] = None,
) -> None:
    # Convert config if necessary
    if isinstance(config, dict):
        config = dict_to_namespace(config)
    config = cast(DiffTrajConfig, config)

    #  Create Model Name if not provided.
    model_name = config.name or generate_model_name(config)

    log.info("#" * 80)
    log.info("#" * 80)

    # Set up the model if not provided
    module = module or get_difftraj_model(config)
    logc(f"Using conditional model:", f"{config.conditional}")
    logc("Using DP-SGD:", f"{config.dp.dp}")

    # Set up the device (read from config if not provided)
    device = get_device(config.gpu)
    module.to(device=device)

    logc(f"Using dataset:", f"{config.data.dataset}")
    if not dataloader:
        dataset = get_difftraj_dataset(config.data.dataset, conditional=config.conditional)
        config.training.batch_size = min(config.training.batch_size, len(dataset))
        # Create DataLoader (no collate_fn needed - dataset is stored in preprocessed form)
        dataloader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True, num_workers=8,
                                drop_last=True, pin_memory=True)
        # Dataloader returns a tuple of (batch, attr) with shapes (batch_size, 2, traj_length) and
        # (batch_size, attr_dim) where attr_dim is the number of conditional attributes
    logc(f"Using Trajectory Length:", f"{config.data.traj_length}")
    logc(f"Using batch size:", f"{dataloader.batch_size}")

    # Foursquare dataset requires a reduced physical batch size
    if config.data.dataset == 'fs':
        max_physical_batch_size = 384
    else:
        global MAX_PHYSICAL_BATCH_SIZE
        max_physical_batch_size = MAX_PHYSICAL_BATCH_SIZE
    del MAX_PHYSICAL_BATCH_SIZE # Remove global variable

    # Training parameters
    n_diff_steps = config.diffusion.num_diffusion_timesteps  # Number of diffusion steps
    # Use a linear schedule for beta
    beta = torch.linspace(config.diffusion.beta_start,
                          config.diffusion.beta_end,
                          n_diff_steps).to(device=device)
    alpha = 1. - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    # Optimizer
    assert config.optimizer.optimizer_type == Optimizer.ADAMW, "Only AdamW is supported for DiffTraj."
    optimizer = torch.optim.AdamW(module.parameters(), lr=config.optimizer.lr)  # Optimizer
    logc(f"Using learning rate:", f"{config.optimizer.lr}")

    # Compute number of epochs based on number of iterations
    config.training.n_iters, config.training.n_epochs = compute_steps_and_epochs(
        steps=config.training.n_iters,
        epochs=config.training.n_epochs,
        steps_per_epoch=len(dataloader))
    n_epochs = config.training.n_epochs

    # Initialize DP
    model = DP_Wrapper(name=model_name, model=module, optimizer=optimizer, dp=config.dp.dp or config.dp.clip_search)
    # Make sure module and optimizer cannot be accessed directly (only via model)
    del module
    del optimizer


    if config.dp.clip_search:
        log.warning("Using ClipSearch - No noise will be added!")
        logc("Clipping Norm:", f"{config.dp.max_grad_norm}")
        dataloader = model.init_dp(
            dataloader=dataloader,
            epochs=n_epochs,
            max_grad_norm=config.dp.max_grad_norm,
            noise_multiplier=0.0,
        )
    elif config.dp.dp:
        dataloader = model.init_dp(
            dataloader=dataloader,
            epochs=n_epochs,
            max_grad_norm=config.dp.max_grad_norm,
            target_epsilon=config.dp.target_epsilon,
            delta=config.dp.target_delta,
        )

    # EMA
    if config.ema:
        logc("EMA:", f"Activated with rate {config.ema_rate}")
        ema_helper = EMAHelper(mu=config.ema_rate, module=model)
    else:
        logc("EMA:", f"Deactivated")
        ema_helper = None

    # Create Tensorboard writer
    writer = get_tensorboard_writer(model_name, use_tensorboard=config.tensorboard)

    # Write config to tensorboard as metadata (pretty print)
    writer.add_text("Config", pprint.pformat(asdict(config)))

    # Determine the number of physical batches per real batch (Without DP physical batch size is equal to batch size)
    n_phy_batch = math.ceil(config.training.batch_size / max_physical_batch_size) if config.dp.dp else 1

    start_time = datetime.now()
    step = 0
    filter_opacus_warnings()
    for epoch in tqdm(range(1, n_epochs + 1), desc="Epochs", leave=True):
        # Note that the batches shown by tqdm are the physical batches not the real batches
        # The physical batch size has been determined experimentally
        with (
                BatchMemoryManager(
                    data_loader=dataloader,
                    max_physical_batch_size=max_physical_batch_size,
                    optimizer=model.opt
                ) if (
                        model.dp_initialised and max_physical_batch_size < config.training.batch_size
                ) else (
                        nullcontext(dataloader)
                )
        ) as new_dataloader:
            for _, batch in tqdm(enumerate(new_dataloader), desc="Batches",
                                          leave=False, total=len(new_dataloader)):
                step += 1
                # Note that the actual gradient steps are only done at the end of each batch size,
                # not after each physical batch!
                real_step = step // n_phy_batch

                # Move to device
                x0, attr = split_batch(batch, config=config, device=device, dtype=torch.float32)

                # Zero the gradients and set model to training mode
                model.train()
                model.opt.zero_grad()

                # Sample the diffusion steps
                t = torch.randint(low=0, high=n_diff_steps, size=(len(x0) // 2 + 1,)).to(device=device)
                # Shape of t is (batch_size // 2 + 1,)
                t = torch.cat([t, n_diff_steps - t - 1], dim=0)[:len(x0)]
                # Make symmetric -->  Shape of t is now (batch_size, )

                # Get the noised images (xt) and the noise (our target)
                xt, noise = q_xt_x0(x0, t, alpha_bar=alpha_bar)
                # Run xt through the network to get its predictions
                if config.conditional:
                    pred_noise = model(xt, t, attr)
                else:
                    # Ensure that the unconditional model is not passed any attributes
                    pred_noise = model(xt, t)
                # Compare the predictions with the targets
                loss = F.mse_loss(noise, pred_noise)
                loss.backward()

                ####################################################################
                # Logging (Needs to be done between the backward and the step)
                ####################################################################
                if step % n_phy_batch == 0:
                    writer.add_scalar("Loss/Loss", loss.item(), real_step)
                    if config.grad_norm_clip > 0 and not config.dp.dp and not config.dp.clip_search:
                        # Clip the gradient norm (Both DP and ClipSearch have their own clipping)
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_norm_clip)
                        writer.add_scalar("Grad/GradNorm", grad_norm, real_step)

                    # Track individual parameter gradients for debugging
                    model.track_gradients(writer=writer, step=real_step)

                # Perform the actual gradient step
                model.opt.step()
                if ema_helper:
                    # Note that this only updates the shadow variables, not the actual model!
                    ema_helper.update(model)

                # Save model every X steps (and only after a full real step!)
                if is_validation_step(step, real_step, n_phy_batch, config.training.save_freq):
                    with logging_redirect_tqdm():
                        model.save_parameters(epoch=epoch, ema_helper=ema_helper)

                # More frequent validation
                if is_validation_step(step, real_step, n_phy_batch, 100):
                    # Do another fwd pass in eval mode to deactivate things like the noise schedule
                    model.eval()
                    with torch.no_grad():
                        pred_noise = model(xt, t, attr)
                        # Compare the predictions with the targets
                        val_loss = F.mse_loss(noise, pred_noise)
                        # Write loss to tensorboard
                        writer.add_scalar("Loss/Validation", val_loss.item(), real_step)

                # Regular Validation
                if is_validation_step(step, real_step, n_phy_batch, config.training.validation_freq):
                    # Visualize the predictions every X epochs
                    # 1. Show the original trajectories
                    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                    # noinspection PyUnboundLocalVariable
                    points_orig = x0.transpose(1, 2).reshape(-1, 2).detach().cpu().numpy()
                    ax.scatter(points_orig[:, 0], points_orig[:, 1], color='blue', alpha=0.1, s=0.1)
                    writer.add_figure("Original", fig, global_step=real_step)
                    # 2. Show the predictions
                    n_samples = min(1000, len(x0))  # If conditional, len(attr) is used.
                    traj_syn = generate_diffTraj(model, config, n_total=n_samples, conditional_attr=attr)
                    fig, ax = cast(Tuple[plt.Figure, plt.Axes], plot_trajectories(traj_syn, connected=False))
                    writer.add_figure("Generated", fig, global_step=real_step)
                    # 3. Add some metrics to tensorboard
                    points_syn = np.array(traj_syn).transpose(0, 2, 1).reshape(-1, 2)
                    points_orig = points_orig[:len(points_syn)]
                    swd = sliced_wasserstein_distance(points_orig, points_syn, seed=config.seed)
                    writer.add_scalar("Metrics/SWD", swd, real_step)
                    jsd = JS_from_points(points_orig, points_syn, grid_size=32, bbox=NORMALISED_BBOX)
                    writer.add_scalar("Metrics/JSD", jsd, real_step)
                # End of batch loop
            # End of BMM with statement

        # Write epsilon to tensorboard only once per epoch b/c it is quite slow
        if config.dp.dp and real_step > 2:
            # Epsilon can only be computed after the first gradient steps
            writer.add_scalar("DP/Epsilon", model.get_epsilon(), real_step)
        # End of epoch loop

    # Save model at end of training
    with logging_redirect_tqdm():
        model.save_parameters(epoch=n_epochs, ema_helper=ema_helper)
    logc("Total Training Duration:", time_to_str(datetime.now() - start_time))

    # Write an error message if epsilon is not below the target
    if config.dp.dp and model.get_epsilon() > config.dp.target_epsilon:
        log.error(f"Failed to reach target epsilon of {config.dp.target_epsilon}! Got {model.get_epsilon()} instead.")


if __name__ == '__main__':
    # Configure logging
    log = configure_root_loger(logging_level=logging.INFO, file=os.path.join(base_config.LOG_DIR, "train_diffTraj.log"))

    # Parse the config
    config = parse(DiffTrajConfig, description="Train a DiffTraj model.")

    # Train the model
    train_diffTraj(config=config)
