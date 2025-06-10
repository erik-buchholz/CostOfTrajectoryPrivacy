#!/usr/bin/env python3
"""Base Configuration."""
import copy
# from dataclasses import field
from typing import Union, Optional, List, Sequence, cast, Any, TypeVar

from pydantic import Field
from pydantic.dataclasses import dataclass
from simple_parsing import field

from privtrajgen.config.enums import *
from privtrajgen.datasets import trajectory_lengths, number_of_trajectories
from privtrajgen.utils import compute_delta


T = TypeVar("T", bound="Config")


@dataclass
class Config:
    def copy(self: T) -> T:
        """Create a 'safe' copy, i.e., any changes to the new instance will not affect the original instance."""
        return copy.deepcopy(self)

    def replace(self: T, **kwargs: Any) -> T:
        """Create a new instance with the given attributes replaced."""
        copied = self.copy()
        for key, value in kwargs.items():
            setattr(copied, key, value)
        return copied


@dataclass(config=dict(validate_assignment=True))
class DPConfig(Config):
    dp: bool  # Whether to use Differential Privacy
    target_delta: Union[float, None] = None  # Target Delta
    target_epsilon: float = 10.0  # Target Epsilon
    max_grad_norm: float = 0.1  # Maximum Gradient Norm
    dp_accountant: str = 'prv'  # Differential Privacy Accountant
    clip_search: bool = False  # Whether to search for the best clipping norm

    # Create a property for adaptive_grad_norm to ensure it's always False - Not Implemented Yet
    @property
    def adaptive_grad_norm(self) -> bool:
        return False


@dataclass(config=dict(validate_assignment=True))
class OptimizerConfig(Config):
    optimizer_type: Optimizer = Optimizer.AUTO  # Optimizer Type
    lr: Optional[float] = None  # Learning Rate
    beta1: Optional[float] = None  # Beta1 for Adam Optimizer - None for Optimizer Defaults
    beta2: Optional[float] = 0.999  # Beta2 for Adam Optimizer - None for Optimizer Defaults
    weight_decay: Optional[float] = None  # Weight Decay - None for Optimizer Defaults

    def __str__(self) -> str:
        txt = f"{self.optimizer_type}({self.lr})"
        if self.beta1 is not None:
            txt += f"({self.beta1}, {self.beta2})"
        if self.weight_decay is not None:
            txt += f"({self.weight_decay})"
        return txt


# Data Configuration is frozen because we don't want to change it after creation
# It is only used as a compatibility proxy.
@dataclass(frozen=True)
class DataConfig:
    traj_length: int
    dataset: Datasets = field(metadata={"custom_args": {"metavar": None}})  # Dataset Name


class TrainingProxy:
    """Compatibility Proxy for Training Configuration of DiffTraj."""

    def __init__(self, config: 'BaseConfig'):
        self.config = config

    @property
    def n_epochs(self) -> Optional[int]:
        return self.config.epochs

    @n_epochs.setter
    def n_epochs(self, value: int) -> None:
        self.config.epochs = value

    @property
    def n_iters(self) -> Optional[int]:
        return self.config.steps

    @n_iters.setter
    def n_iters(self, value: int) -> None:
        self.config.steps = value

    @property
    def batch_size(self) -> int:
        return self.config.batch_size

    @batch_size.setter
    def batch_size(self, value: int) -> None:
        self.config.batch_size = value

    @property
    def lr(self) -> Optional[float]:
        return self.config.optimizer.lr

    @lr.setter
    def lr(self, value: Optional[float]) -> None:
        self.config.optimizer.lr = value

    @property
    def snapshot_freq(self) -> int:
        return self.config.save_freq

    @snapshot_freq.setter
    def snapshot_freq(self, value: int) -> None:
        self.config.save_freq = value

    @property
    def save_freq(self) -> int:
        return self.config.save_freq

    @save_freq.setter
    def save_freq(self, value: int) -> None:
        self.config.save_freq = value

    @property
    def validation_freq(self) -> int:
        return self.config.plot_freq

    @validation_freq.setter
    def validation_freq(self, value: int) -> None:
        self.config.plot_freq = value


@dataclass(config=dict(validate_assignment=True))
class ConditionalInfoConfig(Config):
    """Configuration for Conditional Information."""
    dp: bool = False  # Whether to use Differential Privacy for Conditional Information
    clip_norm: bool = False  # Whether to clip the norm of the conditional information even when not using DP.
    # Ignored if dp is True as clipping is always used in that case.

    source: ConditionalInfoSource = ConditionalInfoSource.DIFFTRAJ  # Source of the Conditional Information

    clipping_type: ConditionalInfoClippingType = ConditionalInfoClippingType.NORM_CLIPPING  # Clipping Type
    max_norm: float = 1.0  # Maximum Norm for Clipping

    noise: NoiseType = NoiseType.GAUSSIAN  # Noise Type --> Determines the norm to use
    location: ConditionalInfoLocation = ConditionalInfoLocation.HIDDEN  # Where to clip / add noise

    epsilon: float = 1.0  # Epsilon for Differential Privacy
    delta: Optional[float] = None  # Delta for Differential Privacy. None for automatic computation.
    dataset: Optional[Datasets] = None  # Optional Dataset to determine delta

    embedding: ConditionalInfoEmbeddingType = ConditionalInfoEmbeddingType.INTEGRATED_FC  # Compression Type

    # Other Options
    increase_noise: bool = False  # Whether to start without noise and gradually converge towards the target noise
    noise_schedule: bool = True  # Whether to use a noise schedule similar to diffusion

    # The following parameters are the standard DiffTraj parameters use for the 8-dim conditional information
    continuous_idx: Sequence[int] = tuple(range(1, 6))  # List of Continuous Attributes
    categorical_idx: Sequence[int] = (0, 6, 7)  # List of Categorical Attributes
    categorical_dims: Sequence[int] = (288, 257, 257)  # Dimensions of Categorical Attributes
    embedding_dim: int = 512  # Dimension of the Embedding Space
    hidden_dim: int = 256  # Dimension of the Hidden Layer

    def __str__(self) -> str:
        txt = f"CI({self.hidden_dim}-{self.embedding_dim})"
        txt += f"-{self.source}"
        if self.dp or self.clip_norm:
            txt += f"-{self.clipping_type}-{self.max_norm}"
            txt += f"-{self.location}"
        if self.dp:
            txt += f"-{self.noise}-{self.epsilon}eps-DP"
            if self.increase_noise:
                txt += "-Inc"
            if self.noise_schedule:
                txt += "-SCHD"
        return txt

    def __post_init__(self) -> None:
        # Verify that the dimensions of categorical attributes match the indices
        if len(self.categorical_idx) != len(self.categorical_dims):
            raise ValueError("The length of categorical_idx and categorical_dims must be the same.")

        # Verify that indices have the right format
        if not isinstance(self.continuous_idx, (list, slice, tuple)):
            raise ValueError("continuous_idx must be a list, slice, or tuple.")
        if not isinstance(self.categorical_idx, (list, slice, tuple)):
            raise ValueError("categorical_idx must be a list, slice, or tuple.")

        # Verify that categorical and continuous indices are disjoint and cover all attributes
        if len(set(self.continuous_idx).intersection(set(self.categorical_idx))) > 0:
            raise ValueError("Continuous and Categorical indices must be disjoint.")
        if len(set(self.continuous_idx).union(set(self.categorical_idx))) != max(max(self.continuous_idx), max(self.categorical_idx)) + 1:
            raise ValueError("Continuous and Categorical indices must cover all attributes.")

        # Auto determine delta if not set
        if self.delta is None:
            if self.noise == NoiseType.LAPLACE or self.noise == NoiseType.VMF:
                self.delta = 0.0
            elif self.dataset is not None:
                self.delta = compute_delta(number_of_trajectories[self.dataset])
            elif self.dp:
                raise ValueError("Delta must be set for Gaussian noise.")
            else:
                self.delta = 0.0  # Not using DP
        # Cast delta to float
        self.delta = cast(float, self.delta)

        # If we use Gaussian noise, delta must not be 0
        if self.dp and self.noise == NoiseType.GAUSSIAN and self.delta == 0:
            raise ValueError("Delta must not be 0 when using Gaussian noise.")

        # For VMF and Laplace noise, delta must be 0
        if self.noise == NoiseType.VMF or self.noise == NoiseType.LAPLACE:
            if self.delta != 0:
                raise ValueError("Delta must be 0 for VMF and Laplace noise.")

        # Currently only integrated FC is implemented as embedding type
        if self.embedding != ConditionalInfoEmbeddingType.INTEGRATED_FC:
            raise NotImplementedError("Only integrated FC is implemented as embedding type.")

        # Ensure that HIDDEN is used for location to ensure compatibility with the description in the paper
        # Comment this line if you want to run your own evaluations.
        if self.location != ConditionalInfoLocation.HIDDEN:
            raise ValueError("HIDDEN location used in paper.")

        # If DP, set clip_norm to True
        if self.dp:
            self.clip_norm = True

        # Check that increase_noise and noise_schedule are not used together
        if self.increase_noise and self.noise_schedule:
            raise ValueError("increase_noise and noise_schedule cannot be used together.")

    @property
    def norm_type(self) -> float:
        if self.noise == NoiseType.GAUSSIAN or self.noise == NoiseType.VMF:
            return 2.0
        elif self.noise == NoiseType.LAPLACE:
            return 1.0
        else:
            raise ValueError(f"Unknown noise type: {self.noise}")


@dataclass(config=dict(validate_assignment=True))
class ValidationConfig(Config):
    """Configuration for Validation."""
    ema: bool = False  # Whether to use EMA parameters for validation
    epoch: Optional[int] = None  # Epoch to validate - Defaults to last epoch
    debug: bool = False  # Whether to use debug mode for validation
    runs: int = 1  # Number of runs for validation
    cond: Optional[ConditionalInfoConfig] = None
    """Conditional Information Configuration for Validation. If None, the training configuration is used."""


@dataclass(config=dict(validate_assignment=True))
class BaseConfig(Config):
    dataset: Datasets = field(metadata={"custom_args": {"metavar": None}})  # Dataset to use
    model_type: Models = field(metadata={"custom_args": {"metavar": None}})  # Model to use
    conditional: bool = False  # Whether the model uses conditional information
    cond: ConditionalInfoConfig = field(default_factory=ConditionalInfoConfig)  # Conditional Information Configuration

    # Architecture Config
    in_channels: int = 2  # Number of Input Channels

    name: str = ""  # Name of the experiment / model
    gpu: int = field(default=0, alias=['-g'])  # GPU ID

    # Experiment Configuration
    seed: int = field(default=42)  # Random Seed
    num_runs: int = 1  # Number of runs
    run_id: int = 1  # Run ID - Set larger than 1 to continue a run. Will begin with the run ID. Must be in [1, num_runs]
    k_fold: int = 5  # K-Fold Cross-Validation

    batch_size: int = 1024  # Batch Size
    steps: Optional[int] = 20000  # Number of training steps (takes precedence over epochs)
    epochs: Optional[int] = None  # Number of training epochs

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)  # Optimizer Configuration
    ema: bool = False  # Whether to use Exponential Moving Average
    ema_rate: float = 0.999  # EMA Rate
    grad_norm_clip: float = 0.0  # Gradient Clipping. 0.0 for no clipping.

    plot_freq: int = 1000  # Plotting Frequency in Steps
    save_freq: int = 10000  # Saving Frequency in Steps
    tensorboard: bool = True  # Whether to use TensorBoard (deactivate for debugging)

    validation: ValidationConfig = field(default_factory=ValidationConfig)  # Validation Configuration

    dp: DPConfig = field(default_factory=lambda: DPConfig(dp=False))  # Differential Privacy Configuration

    def __post_init__(self) -> None:
        # Ensure only one of steps or epochs is set
        if (self.steps is None) == (self.epochs is None):
            raise ValueError("Either steps or epochs must be set, not both or none")
        # Set delta for conditional information based on the dataset if Gaussian, and it has not been set
        if self.cond.noise == NoiseType.GAUSSIAN and self.cond.delta == 0:
            self.cond.delta = compute_delta(number_of_trajectories[self.dataset])
        # Determine name if not set
        if self.name == "":
            pass  # Doesn't work due to circular imports.
            # from privtrajgen.models.naming import generate_model_name
            # self.name = generate_model_name(self)
        # Raise an error if K-Fold is != 5 for conditional models. Ensures consistency throughout eval.
        if self.k_fold != 5 and self.conditional:
            raise ValueError("K-Fold must be 5.")

    @property
    def data(self) -> DataConfig:
        """For compatibility with old DiffTraj configs"""
        return DataConfig(traj_length=trajectory_lengths[self.dataset], dataset=self.dataset)

    @property
    def training(self) -> TrainingProxy:
        return TrainingProxy(config=self)  # Ensures a new instance of TrainingProxy

    @property
    def is_gan(self) -> bool:
        return self.model_type in {Models.UNetGAN}

    @property
    def is_diffusion(self) -> bool:
        return self.model_type in {Models.DiffTraj}


@dataclass(config=dict(validate_assignment=True))
class LRSchedulerConfig(Config):
    scheduler_type: LRSchedulerType = field(LRSchedulerType.NONE,
                                            metadata={"custom_args": {"metavar": None}})  # Scheduler Type
    milestones: Optional[Union[List[int], dict]] = None  # For MultiStepLR
    gamma: float = 0.1  # For MultiStepLR

    def __post_init__(self) -> None:
        if self.scheduler_type == LRSchedulerType.MULTI_STEP and self.milestones is None:
            raise ValueError("Milestones must be set for MultiStepLR")

    def __str__(self) -> str:
        # Generate a string representation that can be used in the get_lr_scheduler function
        if self.scheduler_type == LRSchedulerType.MULTI_STEP:
            return f"{self.scheduler_type}({self.milestones}, {self.gamma})"
        else:
            return f"{self.scheduler_type}"


@dataclass(config=dict(validate_assignment=True))
class KLAnnealingConfig(Config):
    """KL Annealing Configuration."""
    # Annealing Type
    anneal_type: AnnealingType = field(default=AnnealingType.CYCLICAL, metadata={"custom_args": {"metavar": None}})
    klw_min: float = 0.0  # Minimum KL Weight
    klw_max: float = 0.1  # Maximum KL Weight

    # Custom Print Function
    def __str__(self) -> str:
        return f"{self.anneal_type}({self.klw_min}-{self.klw_max})"


@dataclass(config=dict(validate_assignment=True))
class VAEConfig(BaseConfig):
    """Configuration for VAE models."""
    kl_annealing: KLAnnealingConfig = Field(default=KLAnnealingConfig())  # KL Annealing Configuration
    lr_scheduler: LRSchedulerConfig = Field(default=LRSchedulerConfig())  # Learning Rate Scheduler Configuration
    warmup_steps: int = 2000  # Number of warmup steps


@dataclass(config=dict(validate_assignment=True))
class UNetVAEConfig(VAEConfig):
    """Configuration for UNetVAE models."""
    model_type: Models = field(Models.UNetVAE, metadata={"custom_args": {"metavar": None}}, cmd=False)  # Model to use
    optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(
        lr=2e-4, optimizer_type=Optimizer.ADAMW, beta1=0.5))  # Optimizer Configuration
    hidden_dim: int = 128  # Dimension of the Hidden Layer
    out_channels: int = 2  # Number of Output Channels
