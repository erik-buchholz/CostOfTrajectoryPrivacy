#!/usr/bin/env python3
"""Utilities for Machine Learning"""
import contextlib
import logging
import os
import warnings
from datetime import datetime
from typing import List, Optional, Union, Tuple, Dict, Any, cast, Callable

import math
import numpy as np
import torch
import torch.nn
from IPython import display
from matplotlib import pyplot as plt
from pydantic import validate_call
from torch import autograd, nn, Tensor
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from privtrajgen import config
from privtrajgen.config.enums import RnnType, Optimizer, MergeMode, LossCombination, AnnealingType, LRSchedulerType
from privtrajgen.evaluation.config import OptimizerConfig, GANLossConfig, LRSchedulerConfig
from privtrajgen.utils import visualise

log = logging.getLogger()

RNN_TYPES = {
    RnnType.RNN: torch.nn.RNN,
    RnnType.LSTM: torch.nn.LSTM,
    RnnType.GRU: torch.nn.GRU
}

RNN_CELL_TYPES = {
    RnnType.RNN: torch.nn.RNNCell,
    RnnType.LSTM: torch.nn.LSTMCell,
    RnnType.GRU: torch.nn.GRUCell
}

OPTIMIZERS = {
    Optimizer.AUTO: None,
    Optimizer.ADAM: torch.optim.Adam,
    Optimizer.RMSPROP: torch.optim.RMSprop,
    Optimizer.ADAMW: torch.optim.AdamW,
    Optimizer.SGD: torch.optim.SGD,
    Optimizer.RADAM: torch.optim.RAdam,
}

MOMENTUM = {
    Optimizer.ADAM: True,
    Optimizer.RMSPROP: False,
    Optimizer.ADAMW: True,
    Optimizer.SGD: False,
    Optimizer.RADAM: True
}


def logc(header: str, *args: Any, column_width: int = 40) -> None:
    log.info(f"{header:<{column_width}}" + "".join([str(arg) for arg in args]))


@validate_call
def get_rnn(rnn_type: RnnType, input_size: int, hidden_size: int, num_layers: int = 1, dropout: float = 0,
            bidirectional: bool = False) -> torch.nn.Module:
    """
    Interface to get RNN layer for quick exchange in models
    :param rnn_type: type € [rnn, lstm, gru]
    :param input_size:
    :param hidden_size:
    :param num_layers: [Default: 1]
    :param dropout: probability € [0,1] [Default: 0]
    :param bidirectional: Use bidirectional RNN [Default: False]
    :return:
    """

    rnn_kwargs = dict(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=True,
        dropout=dropout,
        bidirectional=bidirectional
    )

    return RNN_TYPES[rnn_type](**rnn_kwargs)


# Set arbitrary_types_allowed=True b/c of nn.Module
@validate_call(config=dict(arbitrary_types_allowed=True))
def get_optimizer(
        parameters: Any,
        optimizer: Union[Optimizer, OptimizerConfig] = OptimizerConfig(),
        lr: Optional[float] = None,
        beta1: Optional[float] = None,
        beta2: Optional[float] = None,
        weight_decay: Optional[float] = None,
        loss: Optional[GANLossConfig] = GANLossConfig(False, False, False, "GAN"),
        wgan: Optional[bool] = None,
        gradient_penalty: Optional[bool] = None,
) -> torch.optim.Optimizer:
    """
    Interface to get optimizer for quick exchange in models
    :param parameters: Model parameters
    :param optimizer: type € [auto, adam, rmsprop, adamw, sgd]
    :param lr: Learning rate
    :param beta1: Beta 1 (Adam only)
    :param beta2: Beta 2 (Adam only)
    :param weight_decay: Weight decay
    :param loss: GAN Loss Configuration
    :param wgan: Wasserstein GAN
    :param gradient_penalty: Gradient Penalty
    :return:
    """
    # Unpack config if necessary
    # However, explicit values take precedence over config for backward compatibility
    if isinstance(optimizer, OptimizerConfig):
        optimizer_type = optimizer.optimizer_type
        lr = lr or optimizer.lr
        beta1 = beta1 or optimizer.beta1
        beta2 = beta2 or optimizer.beta2
        weight_decay = weight_decay or optimizer.weight_decay
    else:
        optimizer_type = optimizer

    # Unpack loss config
    if wgan is None and loss is not None:
        wgan = loss.wgan
    else:
        raise ValueError("WGAN flag must be set explicitly or in the loss config.")
    if gradient_penalty is None and loss is not None:
        gradient_penalty = loss.gradient_penalty
    else:
        raise ValueError("Gradient Penalty flag must be set explicitly or in the loss config.")

    # Make sure loss and optimizer objects cannot be accessed anymore to avoid confusion
    del loss
    del optimizer

    if optimizer_type != Optimizer.AUTO and optimizer_type not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer type: {str(optimizer_type)}.")

    # Resolve automatic mode
    if optimizer_type == Optimizer.AUTO:
        if wgan and not gradient_penalty:
            # Standard WGAN without gradient penalty
            optimizer_type = Optimizer.RMSPROP
        else:
            # All other cases
            optimizer_type = Optimizer.ADAMW

    kwargs: Dict[str, Any] = dict()
    if optimizer_type == Optimizer.RADAM:
        kwargs['decoupled_weight_decay'] = True
        # Set weight decay to 0.01 if not set explicitly
        weight_decay = weight_decay or 0.01

    if weight_decay is not None:
        kwargs['weight_decay'] = weight_decay

    if MOMENTUM[optimizer_type] and beta1 is not None and beta2 is not None:
        kwargs['betas'] = (beta1, beta2)

    opt: torch.optim.Optimizer = OPTIMIZERS[optimizer_type](parameters, lr=lr, **kwargs)

    # Log optimizer choice
    log.info(f"Using optimizer: {optimizer_type} with lr={lr}, betas=({beta1}, {beta2}), "
             f"weight_decay={weight_decay}")

    return opt


@validate_call
def get_rnn_cell(rnn_type: RnnType, input_size: int, hidden_size: int) -> torch.nn.Module:
    """
    Interface to get RNN cell for quick exchange in models
    :param rnn_type: type € [rnn, lstm, gru]
    :param input_size:
    :param hidden_size:
    :return:
    """

    rnn_kwargs = dict(
        input_size=input_size,
        hidden_size=hidden_size
    )

    return RNN_CELL_TYPES[rnn_type](**rnn_kwargs)


def l1_regularizer(weights: Tensor, l1: float = 0.01) -> Tensor:
    """PyTorch replacement for keras.regularizers.l1

    :return: L1 penalty term
    """
    return l1 * torch.norm(weights, p=1)


def compute_gradient_penalty(
        discriminator: nn.Module,
        real: Tensor,
        synthetic: Tensor,
        lp: bool = False,
        *args: Any,
        **kwargs: Any
) -> Tensor:
    """
    Implementation of gradient penalty is based on [2]. Lipschitz penalty added by me based on [1].

    Usually, real and synthetic samples are provided as (batch_size, sequence_length, feature_dim) tensors.
    However, some models, particularly the LSTM-TrajGAN-based models, provide a list of tensors with
    (batch_size, sequence_length, feature_dim) tensors for each feature.

    [1] H. Petzka, A. Fischer, and D. Lukovnicov, “On the regularization of Wasserstein GANs.” arXiv, Mar. 05, 2018.
    doi: 10.48550/arXiv.1709.08894.
    [2] A. Sankar, “Demystified: Wasserstein GAN with Gradient Penalty,” Medium. Accessed: May 02, 2023. [Online].
    Available: https://towardsdatascience.com/demystified-wasserstein-gan-with-gradient-penalty-ba5e9b905ead

    :param discriminator: torch.nn.Module
    :param real: (batch_size, sequence_length, feature_dim) or (num_features, batch_size, sequence_length, feature_dim)
    :param synthetic: (batch_size, sequence_length, feature_dim) or (num_features, batch_size, sequence_length, feature_dim)
    :param lp: Use Lipschitz penalty [1] instead of gradient penalty.
    :param args: Additional arguments for the discriminator
    :param kwargs: Additional keyword arguments for the discriminator (e.g., lengths)
    :return: Gradient/Lipschitz Penalty
    """
    device = real.device if isinstance(real, Tensor) else real[0].device

    # Random weight term for interpolation between real and fake samples

    # Differentiate between single and multi-feature input
    if isinstance(real, Tensor) and isinstance(synthetic, Tensor):
        # Trajectories provided as (batch_size, sequence_length, feature_dim) tensors
        batch_size = len(real)
        alpha = torch.rand(batch_size, 1, 1).to(device=device)
        alpha = alpha.expand_as(real)
        # Get random interpolation between real and fake samples
        interpolation = alpha * real + (1 - alpha) * synthetic
        interpolates = autograd.Variable(interpolation, requires_grad=True).to(device=device)
    elif isinstance(real, list) and isinstance(synthetic, list):
        batch_size = len(real[0])
        alpha = torch.rand(batch_size, 1, 1).to(device)
        interpolates = []
        for i, feature in enumerate(real):
            alpha_expanded = alpha.expand_as(feature)
            interpolation = alpha_expanded * feature + (1 - alpha_expanded) * synthetic[i]
            interpolates.append(autograd.Variable(interpolation, requires_grad=True).to(device))
    else:
        raise ValueError("Real and synthetic samples must be either both tensors or both lists.")

    # Check if the RNN uses cuDNN backend and if so, disable cuDNN for this part of the code
    # Necessary b/c of `NotImplementedError: the derivative for '_cudnn_rnn_backward' is not implemented.`
    if contains_rnn(discriminator):
        if device.type == 'cpu':
            # noinspection PyUnresolvedReferences,PyProtectedMember
            torch._C._set_mkldnn_enabled(False)
            context = contextlib.nullcontext()
        else:
            context = torch.backends.cudnn.flags(enabled=False)
    else:
        context = contextlib.nullcontext()
    with context:
        # Get logits for interpolated samples
        d_interpolates = discriminator(interpolates, *args, **kwargs)
    # If discriminator returns tuple, concatenate
    if isinstance(d_interpolates, tuple):
        d_interpolates = torch.cat(d_interpolates, dim=0)
    grad_outputs = torch.ones_like(d_interpolates).to(device=device)

    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.reshape(gradients.size(0), -1)

    if lp:
        # Lipschitz Penalty
        gradient_penalty = (torch.clamp(gradients.norm(2, dim=1) - 1., min=0, max=None) ** 2).mean()
    else:
        # Gradient Penalty
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    log.debug("Unweighted Gradient Penalty: ", gradient_penalty.item())
    return gradient_penalty


def compute_wdiv_penalty(real: Tensor, fake: Tensor, real_validity: Tensor, fake_validity: Tensor, p: int = 6,
                         k: int = 2) -> Tensor:
    DEVICE = real.device
    real_grad_outputs = torch.ones_like(real_validity, dtype=torch.float32, requires_grad=False, device=DEVICE)
    fake_grad_outputs = torch.ones_like(fake_validity, dtype=torch.float32, requires_grad=False, device=DEVICE)

    real_gradient = torch.autograd.grad(
        outputs=real_validity,
        inputs=real,
        grad_outputs=real_grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    fake_gradient = torch.autograd.grad(
        outputs=fake_validity,
        inputs=fake,
        grad_outputs=fake_grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    real_gradient_norm = real_gradient.reshape(real_gradient.size(0), -1).pow(2).sum(1) ** (p / 2)
    fake_gradient_norm = fake_gradient.reshape(fake_gradient.size(0), -1).pow(2).sum(1) ** (p / 2)

    div_gp = torch.mean(real_gradient_norm + fake_gradient_norm) * k / 2
    return div_gp


def clip_discriminator_weights(dis: nn.Module, clip_value: float) -> None:
    for p in dis.parameters():
        p.data.clamp_(-clip_value, clip_value)


def contains_rnn(model: nn.Module) -> bool:
    return any(isinstance(module, torch.nn.RNNBase) for module in model.modules())


def validate_loss_options(wgan: bool, gradient_penalty: bool, lp: bool, wdiv: bool = False) -> None:
    """
    Validates the combination of loss options for consistency.

    :param wgan: Indicates if Wasserstein GAN is used.
    :param gradient_penalty: Indicates if Gradient Penalty is used.
    :param lp: Indicates if Lipschitz Penalty is used.
    :param wdiv: Indicates if Wasserstein Divergence is used.
    :return: None
    :raises ValueError: If an invalid combination of options is provided.
    """
    if wdiv and gradient_penalty:
        raise ValueError("Either Wasserstein Divergence OR Gradient Penalty is possible, not both.")
    if gradient_penalty and not wgan:
        raise ValueError("Gradient Penalty only works with Wasserstein Loss.")
    if wdiv and not wgan:
        raise ValueError("Wasserstein Divergence only works with Wasserstein Loss.")
    if lp and not gradient_penalty:
        raise ValueError("Lipschitz Penalty only works with Gradient Penalty.")


def verify_non_linearity(wgan: bool, last_layer: torch.nn.Module) -> None:
    """
    Verifies that the correct non-linearity is used in the discriminator's last layer.

    :param wgan: Indicates if Wasserstein GAN is used.
    :param last_layer: The last layer of the discriminator model.
    :raises ValueError: If an incorrect non-linearity is used.
    """
    if not wgan:
        if not isinstance(last_layer, torch.nn.Sigmoid):
            raise ValueError("For standard GAN, last layer has to be sigmoid!")
    else:
        if isinstance(last_layer, torch.nn.Sigmoid):
            raise ValueError("For WGAN, no sigmoid is used as last layer!")


class NullWriter:
    """
    A class to replace a Tensorboard SummaryWriter without code modifications.
    """

    def __getattr__(self, _: Any) -> Callable[..., None]:
        # Return a lambda function that does nothing
        return lambda *args, **kwargs: None


def visualize_trajectory_samples(
        gen_samples: torch.Tensor,
        real_samples: torch.Tensor,
        epoch: int,
        batch_i: int,
        real_lengths: Optional[List[int]] = None,
        gen_lengths: Optional[List[int]] = None,
        notebook: bool = False,
        tensorboard: bool = False,
        writer: Optional[Union[SummaryWriter, NullWriter]] = None
) -> None:
    """
    Visualizes trajectory samples by plotting an example trajectory and a point cloud of generated and real samples.

    :param gen_samples: Tensor of generated samples.
    :param real_samples: Tensor of real samples.
    :param gen_lengths: List of lengths of the generated samples if padded.
    :param epoch: Current epoch number for titling the plots.
    :param batch_i: Current batch index for titling the plots.
    :param real_lengths: List of lengths of the real samples if padded. Defaults to None.
    :param notebook: Specifies whether the visualization is in a Jupyter notebook. Defaults to False.
    :param tensorboard: Specifies whether to log the plots to TensorBoard. Defaults to False.
    :param writer: TensorBoard writer object. Required if tensorboard is True.

    :returns: None
    """

    # Convert to numpy
    gen_samples = gen_samples.detach().cpu().numpy()
    if not isinstance(real_samples, list):
        real_samples = real_samples.detach().cpu().numpy()

    if real_lengths is not None:
        # Slice to real length
        real_samples = [r[:real_lengths[i]] for i, r in enumerate(real_samples)]

    if gen_lengths is not None:
        # Slice to real length
        gen_samples = [g[:gen_lengths[i]] for i, g in enumerate(gen_samples)]

    # Randomly select one real and one generated sample for example plot
    idx = torch.randint(low=0, high=len(gen_samples), size=(1,))
    gen_traj = gen_samples[idx]
    real_traj = real_samples[idx]

    # Plot
    if len(gen_traj) > 1:  # Only plot if a sequence
        example_fig, _ = visualise.plot_trajectories(
            [real_traj, gen_traj],
            ['Real', 'Fake'],
            bbox=(-1, 1, -1, 1),  # Points are normalized to [-1, 1]
            title=f"Example Trajectory: Epoch {epoch} - Batch {batch_i}"
        )
    else:
        example_fig = None

    # Plot pointcloud
    gen_points = np.concatenate(gen_samples, axis=0)
    real_points = np.concatenate(real_samples, axis=0)

    heatmap, _ = visualise.plot_pointclouds(
        [gen_points, real_points],
        title=f"Point Cloud: Epoch {epoch} - Batch {batch_i}",
        labels=['Fake', 'Real'],
    )

    if notebook:
        # Display both figures by replacing the old one
        if example_fig is not None:
            display.display(example_fig)
        display.display(heatmap)
        # Close old figures
        plt.close('all')
    if tensorboard:
        if writer is None:
            raise ValueError("TensorBoard writer must be provided if tensorboard logging is enabled.")
        if example_fig is not None:
            writer.add_figure('Example', example_fig, global_step=batch_i)
        writer.add_figure('Heatmap', heatmap, global_step=batch_i)


def visualize_mnist_sequential(
        gen_samples: torch.Tensor,
        batches_done: int,
        notebook: bool = False,
        tensorboard: bool = False,
        writer: Optional[Union[SummaryWriter, NullWriter]] = None,
) -> None:
    gen_samples = gen_samples.view(-1, 1, 28, 28)
    if notebook:
        # Make a grid of images and convert it to numpy for plotting
        grid = make_grid(gen_samples[:25], nrow=5, normalize=True).permute(1, 2, 0).detach().cpu().numpy()
        # Display the grid of images
        fig, ax = plt.subplots()
        ax.imshow(grid)
        ax.axis('off')

        # Show new plot
        display.display(plt.gcf())
        # Close old figure
        plt.close('all')
    if tensorboard and writer is not None:
        # Write to TensorBoard
        writer.add_images("Example", gen_samples.data[:25], batches_done)


def compute_mask(samples: torch.Tensor, padding_value: float = 0.0) -> torch.Tensor:
    """
    Compute a mask for the given samples.

    :param samples: Padded samples with shape (batch_size, time_steps, features).
    :param padding_value: Value used for padding. Defaults to 0.0.
    :returns: Mask with shape (batch_size, time_steps), where each entry is 1 if the corresponding sample is not padding and 0 if it is padding.
    """
    # Assuming padding is done in the time dimension, and the padding value is constant across all features
    mask = (samples != padding_value).any(dim=-1).float()
    return mask


def compute_len_from_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    Compute the length of each sample from the given mask.

    :param mask: Mask with shape (batch_size, time_steps), where each entry is 1 if the corresponding sample is not padding and 0 if it is padding.
    :returns: Length of each sample with shape (batch_size,).
    """
    # Sum over time dimension
    lengths = mask.sum(dim=-1)
    return lengths


def compute_mask_from_lengths(x: torch.Tensor, lengths: List[int] | torch.Tensor) -> torch.Tensor:
    """
    Compute a mask for a given tensor x and a list of lengths.
    :param x: Expected shape = (N, L, C), for N = batch size, L = sequence length, C = feature channels
    :param lengths: List of lengths for each sequence in x
    :return: Binary mask of shape (N, L) with 1s for valid entries and 0s for invalid entries
    """
    # Validate input lengths
    max_length = x.size(1)
    if len(lengths) != x.size(0):
        raise ValueError("The number of lengths must match the batch size of x.")
    if max(lengths) > max_length:
        raise ValueError("Lengths exceed the maximum sequence length of x.")

    # Generate a range tensor that represents sequence positions
    sequence_positions = torch.arange(max_length, device=x.device)

    # Expand the sequence_positions tensor to match the batch size
    expanded_positions = sequence_positions.expand(len(lengths), max_length)

    # Convert lengths to a tensor and reshape for broadcasting
    lengths_tensor = torch.tensor(lengths, device=x.device).unsqueeze(-1)

    # Generate the mask by comparing each position to the corresponding length
    mask = expanded_positions.lt(lengths_tensor)

    return mask


def merge_bilstm_output(x: torch.Tensor, mode: MergeMode) -> torch.Tensor:
    """
    Merge the output of a bidirectional LSTM
    :param x: Output of the LSTM
    :param mode: How to merge the output (sum, mean, concat)
    :return:
    """
    if mode == MergeMode.SUM:
        x = torch.sum(x.view(x.size(0), x.size(1), 2, -1), dim=2)
    elif mode == MergeMode.AVERAGE:
        x = torch.mean(x.view(x.size(0), x.size(1), 2, -1), dim=2)
    elif mode == MergeMode.CONCAT:
        pass  # Default
    elif mode == MergeMode.MUL:
        x = torch.prod(x.view(x.size(0), x.size(1), 2, -1), dim=2)
    else:
        raise ValueError(f"Unknown merge mode: {mode}. Valid modes are 'sum', 'average', 'concat', and 'mul'.")
    return x


def combine_losses(losses: List[torch.Tensor], method: LossCombination = LossCombination.MEAN,
                   weights: Optional[List[float]] = None, step: Optional[int] = None) -> torch.Tensor:
    """
    Combine losses into a single loss value.

    :param losses: List of losses to combine
    :param method: Method to combine losses.
        'mean': Mean of losses
        'sum': Sum of losses
        'max': Maximum of losses
        'min': Minimum of losses
        'prod': Product of losses
        'none': Return list of losses
        'alternate': Alternate between losses
    :param weights: Weights for each loss
    :param step: Global step (Batch number)
    :return: Combined loss
    """
    if weights is None:
        weights = [1.0] * len(losses)
    assert len(losses) == len(weights), "Number of losses and weights must match."
    if method == LossCombination.MEAN:
        return sum([w * l for w, l in zip(weights, losses)]) / sum(weights)
    elif method == LossCombination.SUM:
        return sum([w * l for w, l in zip(weights, losses)])
    elif method == LossCombination.MAX:
        return max(losses)
    elif method == LossCombination.MIN:
        return min(losses)
    elif method == LossCombination.PROD:
        return torch.prod(torch.stack(losses))
    elif method == LossCombination.NONE:
        return losses
    elif method == LossCombination.ALTERNATE:
        if step is None:
            raise ValueError("Step must be provided for alternate loss combination.")
        return losses[step % len(losses)]
    else:
        raise NotImplementedError(f"Method '{method}' not implemented.")


# noinspection PyShadowingBuiltins
def compute_kl_weight(step: int,
                      total_steps: int,
                      num_steps: Optional[int] = None,
                      n_cycles: int = 4,
                      ratio: float = 0.5,
                      min_weight: float = 0,
                      max_weight: float = 1,
                      type: AnnealingType = AnnealingType.LINEAR
                      ) -> float:
    """
    Computes the KL weight based on the specified annealing type over training steps.

    The cyclical annealing schedule is based on https://github.com/haofuml/cyclical_annealing.

    :param step: (int) The current step during training.
    :param total_steps: (int) The total number of steps.
    :param num_steps: (int) The total number of steps to reach the kl_weight_max. Default: total_steps.
    :param n_cycles: (int) The number of cycles to complete during training. Used by cyclical annealing.
    :param ratio: (float) The ratio of the annealing period to the total number of steps. Used by cyclical annealing.
    :param min_weight: (float) The starting value of the KL weight.
    :param max_weight: (float) The maximum value of the KL weight to reach.
    :param type: (AnnealingType) The type of annealing ("constant", "linear", "exponential", "cyclical", or "logistic").
        constant: The KL weight remains constant at the max_weight.
    Returns:
    - float: The computed KL weight for the current step.
    """
    step = max(1, step)

    if step > total_steps:
        # Sometimes the step exceeds the total steps due to rounding errors
        return max_weight

    if min_weight == max_weight or type == AnnealingType.CONSTANT:
        if min_weight != max_weight:
            log.warning(f"Using constant KL Weight of maximum = {max_weight}!.")
        return max_weight
    elif type == AnnealingType.LINEAR:
        n_steps = num_steps or (total_steps / 2)
        if step >= n_steps:
            kl_weight = max_weight
        else:
            kl_weight = min_weight + (max_weight - min_weight) * (step / n_steps)
    elif type == AnnealingType.EXPONENTIAL:
        n_steps = num_steps or total_steps
        steepness = 3.0
        kl_weight = min_weight + (max_weight - min_weight) * (np.exp(steepness * step / n_steps) - 1) / (
                    np.exp(steepness) - 1)
    elif type == AnnealingType.LOGISTIC:
        n_steps = num_steps or total_steps
        midpoint = n_steps / 2
        steepness = 10 / n_steps
        kl_weight = min_weight + (max_weight - min_weight) / (1 + np.exp(-steepness * (step - midpoint)))
    elif type == AnnealingType.CYCLICAL:
        period = total_steps / n_cycles
        mid_point = int(ratio * period)
        if step % period < mid_point:
            kl_weight = min_weight + (max_weight - min_weight) * (step % period) / mid_point
        else:
            kl_weight = max_weight
    else:
        raise ValueError(
            f"Unsupported annealing type {type}. Choose 'linear', 'exponential', 'logistic', 'cyclical', or 'constant'.")

    return kl_weight


def get_lr_scheduler(
        lr_scheduler: Union[List[int], torch.optim.lr_scheduler.LRScheduler, dict, LRSchedulerType, LRSchedulerConfig],
        opt: torch.optim.Optimizer,
        steps: int,
        steps_per_epoch: int,
        ) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Defines and assigns a learning rate scheduler to the model based on the specified scheduler type.

    :param lr_scheduler: The type of learning rate scheduler to be used. It can be a string ('cosine', 'cosRes'),
                         a list of milestones for MultiStepLR, a dictionary of settings for MultiStepLR,
                         or a custom scheduler.
    :param opt: The optimizer object to be used for the scheduler.
    :param steps: The number of steps to be used for the CosineAnnealingLR scheduler.
    :param steps_per_epoch: The number of steps per epoch to be used for the CosineAnnealingWarmRestarts scheduler.
    """
    gamma = 0.1
    if isinstance(lr_scheduler, LRSchedulerConfig):
        # Unpack values
        gamma = lr_scheduler.gamma
        if lr_scheduler.scheduler_type == LRSchedulerType.MULTI_STEP:
            lr_scheduler = lr_scheduler.milestones
        else:
            lr_scheduler = lr_scheduler.scheduler_type
    if lr_scheduler == LRSchedulerType.NONE:
        lr_scheduler = None
        log.info("No learning rate scheduler is used.")
    elif lr_scheduler == LRSchedulerType.COSINE:
        # See: https://github.com/Tony-Y/pytorch_warmup/blob/master/README.md
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
        log.info(f"Using Cosine Annealing LR Scheduler with T_max={steps}")
    elif lr_scheduler == LRSchedulerType.COSRES:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=steps_per_epoch, T_mult=2)
        # We restart the scheduler after each epoch and double the period each time
        # Compare the paper: https://arxiv.org/pdf/1608.03983 Section 3
        log.info(f"Using Cosine Annealing Warm Restarts with T_0={steps_per_epoch} and T_mult=2")
    elif lr_scheduler is not None and isinstance(lr_scheduler, list):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=lr_scheduler, gamma=gamma)
        log.info(f"Using MultiStepLR with milestones={lr_scheduler}")
    elif lr_scheduler is not None and isinstance(lr_scheduler, dict):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, **lr_scheduler)
        log.info(f"Using MultiStepLR with settings: {lr_scheduler}")
    elif isinstance(lr_scheduler, torch.optim.lr_scheduler.LRScheduler):
        lr_scheduler = lr_scheduler
    else:
        raise ValueError(f"Unknown learning rate scheduler type: {lr_scheduler}")
    return lr_scheduler


def compute_steps_and_epochs(steps: Optional[int], epochs: Optional[int], steps_per_epoch: int,
                             precendence: str = "steps") -> Tuple[int, int]:
    """
    Computes the number of steps and epochs based on the provided values.

    :param steps: The number of steps to be used for training.
    :param epochs: The number of epochs to be used for training.
    :param steps_per_epoch: The number of steps per epoch (usually len(dataloader)).
    :param precendence: The precedence of the provided values. Either "steps" or "epochs".
    :return: The number of steps and epochs.
    """
    if steps is None and epochs is None:
        raise ValueError("Either steps or epochs must be set, not both or none.")

    if steps is not None and (epochs is None or precendence == "steps"):
        # Compute epochs from steps
        epochs = int(math.ceil(steps / steps_per_epoch))
    elif epochs is not None and (steps is None or precendence == "epochs"):
        # Compute steps from epochs
        steps = epochs * steps_per_epoch
    else:
        raise ValueError(f"Unknown precedence: {precendence}. Choose 'steps' or 'epochs'.")
    return steps, epochs


def get_tensorboard_writer(model_name: str, use_tensorboard: bool = True) -> SummaryWriter:
    """
    Get a TensorBoard writer for logging training metrics.

    :param model_name: The name of the model to be used for logging.
    :param use_tensorboard: Specifies whether to use TensorBoard for logging. Defaults to True.
    :return: The TensorBoard writer object.
    """
    if use_tensorboard:
        writer = SummaryWriter(
            log_dir=os.path.join(config.TENSORBOARD_DIR, f"{datetime.now().strftime('%b-%d_%X_')}{model_name}/"))
    else:
        writer = NullWriter()
    return writer


def filter_opacus_warnings() -> None:
    """
    Filter out the "Using a non-full backward hook" warning from Opacus.
    It is safe to ignore this warning as it does not affect the training process:
    https://discuss.pytorch.org/t/gan-raises-userwarning-using-a-non-full-backward-hook-when-the-forward-contains-multiple/175638/2
    """
    warnings.filterwarnings("ignore", message="Using a non-full backward hook")


def get_number_of_gpus() -> int:
    return cast(int, torch.cuda.device_count())


def get_device(gpu: Optional[int]) -> torch.device:
    device = torch.device(f"cuda:{gpu}" if (torch.cuda.is_available() and gpu is not None and gpu >= 0) else "cpu")
    logc(f"Using device:", f"{device}")
    return device


@validate_call
def is_validation_step(step: int, real_step: int, ratio: int, freq: int) -> bool:
    if real_step % freq == 0 and step % ratio == 0:
        return True
    return False
