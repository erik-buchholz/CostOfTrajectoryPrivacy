#!/usr/bin/env python3
"""DiffTraj Configuration."""
from types import SimpleNamespace
from typing import Sequence

from pydantic.dataclasses import dataclass
from simple_parsing import field

from privtrajgen.config import BetaSchedule, Models
from privtrajgen.config.enums import Optimizer
from privtrajgen.utils import dict_to_namespace, enum_field
from .baseconfig import BaseConfig, OptimizerConfig


@dataclass(config=dict(validate_assignment=True))
class DiffusionConfig:
    """Diffusion-specific configuration."""
    beta_schedule: BetaSchedule = enum_field(default=BetaSchedule.LINEAR)  # Beta Schedule
    beta_start: float = 0.001
    beta_end: float = 0.02
    num_diffusion_timesteps: int = 500  # Number of Diffusion Timesteps (Training)
    num_generation_timesteps: int = 100  # Number of Diffusion Timesteps (Generation)
    eta: float = 0.0  # Eta for Diffusion (Only used for generation)

@dataclass(config=dict(validate_assignment=True))
class DiffTrajModelConfig:
    """For compatibility with old format."""
    type: str = "simple"
    attr_dim: int = 8
    guidance_scale: int = 3
    in_channels: int = 2
    out_ch: int = 2
    ch: int = 128
    ch_mult: Sequence[int] = (1, 2, 2, 2)
    num_res_blocks: int = 2
    attn_resolutions: Sequence[int] = (16,)
    dropout: float = 0.1
    var_type: str = 'fixedlarge'
    resamp_with_conv: bool = True

def _update_from_namespace(cfg: BaseConfig, namespace: SimpleNamespace, ignore: Sequence = tuple()) -> None:
    """Recursively updates a config from a namespace."""
    for key, value in namespace.__dict__.items():
        if key in ignore:
            continue
        if isinstance(value, SimpleNamespace):
            _update_from_namespace(getattr(cfg, key), value, ignore)
        else:
            setattr(cfg, key, value)

@dataclass(config=dict(validate_assignment=True))
class DiffTrajConfig(BaseConfig):
    """DiffTraj-specific configuration."""
    model_type: Models = field(Models.DiffTraj, metadata={"custom_args": {"metavar": None}}, cmd=False)
    model: DiffTrajModelConfig = field(default_factory=DiffTrajModelConfig)
    diffusion: DiffusionConfig = field(default_factory=lambda: DiffusionConfig(
        beta_schedule=BetaSchedule.LINEAR,
        beta_start=0.0001,
        beta_end=0.05,
        num_diffusion_timesteps=500
    ))
    # Overwrite Default LR
    optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(
        Optimizer.ADAMW, lr=2e-4, beta1=0.9, beta2=0.999))
    """Optimizer configuration."""
    ema_rate: float = 0.999

    # Use Gradient Clipping to avoid exploding gradients (Will only be used if DP-SGD is not used)
    grad_norm_clip: float = 1.0

    def __post_init__(self) -> None:
        assert self.in_channels == self.model.in_channels, "Incompatible in_channels in DiffTrajConfig and DiffTrajModelConfig."

    @classmethod
    def from_legacy_dict(cls, legacy_dict: dict) -> "DiffTrajConfig":
        """Converts a legacy dictionary to a DiffTrajConfig instance."""
        legacy_cfg = dict_to_namespace(legacy_dict)
        # For simplicity, we want to use properties.
        # So we create a config first and set it after.
        # However, config.data is frozen, so we need to initialise it correctly.
        config = DiffTrajConfig(dataset=legacy_cfg.data.dataset)
        _update_from_namespace(config, legacy_cfg, ignore=["data", "ema", "ema_rate", "sampling"])
        return config
