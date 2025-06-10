#!/usr/bin/env python3
"""Naming conventions for models."""
from pydantic import validate_call

from privtrajgen.evaluation.config import BaseConfig, UNetGANConfig, UNetVAEConfig, Config, PrivTraceConfig
from privtrajgen.utils import clean_filename


def format_lr(lr: float) -> str:
    """Bring learning rate into format Xe-Y."""
    txt = f"{lr:.0e}"
    return txt


@validate_call(config={"arbitrary_types_allowed": True})
def generate_model_name(config: PrivTraceConfig | BaseConfig) -> str:
    """Generate a model name from the config"""
    if isinstance(config, PrivTraceConfig):
        return config.name
    else:
        assert isinstance(config, BaseConfig), "Config must be a BaseConfig or derived class"
        # Whether DP is enabled
        name = f"{'DP-' if config.dp.dp else ''}"
        # Model Type
        name += f"{config.model_type.name}-"
        if isinstance(config, UNetGANConfig):
            # Use '', 'WD', 'WG', or 'WDG' to indicate weak discriminator, weak generator, or both
            if not config.strong_dis and not config.strong_gen:
                name += "WDG-"
            elif not config.strong_dis:
                name += "WD-"
            elif not config.strong_gen:
                name += "WG-"
        # Cond or Uncond
        name += f"{'cond' if config.conditional else 'uncond'}"
        # Dataset
        name += f"_{config.dataset.upper()}"
        # Append the # steps to the model name (for 20000 use _20k)
        if config.steps is not None:
            name += f"_{config.steps // 1000}k"
        else:
            name += f"_{config.epochs}e"
        # Learning Rate (different approach for GANs)
        if isinstance(config, UNetGANConfig):
            name += f"_lrd{format_lr(config.lr_dis)}_lrg{format_lr(config.lr_gen)}"
            name += f"_{config.n_critic}x{config.loss.name}"
        elif config.optimizer.lr is not None:
            name += f"_lr{format_lr(config.optimizer.lr)}"
        # For VAE, add KL Annealing
        if isinstance(config, UNetVAEConfig):
            name += f"_LRS-{config.lr_scheduler}_KL-{config.kl_annealing}"
        # Append the clip norm if provided
        if config.dp.dp or config.dp.clip_search:
            name += f"_C{config.dp.max_grad_norm}{'-Ada' if config.dp.adaptive_grad_norm else ''}"
        if config.dp.dp:
            name += f"_eps{config.dp.target_epsilon}"
        # Append representation of conditional information
        if config.conditional:
            name += f"_{str(config.cond)}"
        # Append number of folds if provided
        if config.k_fold > 1:
            name += f"_KF{config.k_fold}"
        # Append run_id if provided in XX format
        if config.run_id is not None:
            name += f"_run{config.run_id:02d}"

        # Replace all symbols that should not be in a filename
        name = clean_filename(name)

        # Update the config with the model name
        config.name = name
        return name
