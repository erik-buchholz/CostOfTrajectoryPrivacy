#!/usr/bin/env python3
"""Define Enums so that they can be used wihtout circular imports."""
from enum import Enum

from .constants import GEOLIFE, PORTO, MNISTSequential, FSNYC


class CustomEnum(str, Enum):
    """If you call str() on a CustomEnum, you get the name instead of Classname.name.
    Moreover, for
    enum = CustomEnum()
    enum == x
    Is true for x == enum.value and x == enum and x == enum.name
    """
    def __str__(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.value == other or self.name == other
        elif isinstance(other, CustomEnum):
            return self.value == other.value
        return False

    def __hash__(self) -> int:
        return hash(self.value)


class RnnType(CustomEnum):
    RNN = 'rnn'
    LSTM = 'lstm'
    GRU = 'gru'

class Optimizer(CustomEnum):
    AUTO = 'auto'
    ADAM = 'adam'
    RMSPROP = 'rmsprop'
    ADAMW = 'adamw'
    SGD = 'sgd'
    RADAM = 'radam'

class MergeMode(CustomEnum):
    SUM = 'sum'
    AVERAGE = 'average'
    CONCAT = 'concat'
    MUL = 'mul'

class LossCombination(CustomEnum):
    MEAN = 'mean'
    SUM = 'sum'
    MAX = 'max'
    MIN = 'min'
    PROD = 'prod'
    NONE = 'none'
    ALTERNATE = 'alternate'

class AnnealingType(CustomEnum):
    LINEAR = 'linear'
    EXPONENTIAL = 'exponential'
    LOGISTIC = 'logistic'
    CYCLICAL = 'cyclical'
    CONSTANT = 'constant'

class Datasets(CustomEnum):
    """Enumeration of datasets."""
    FSNYC = FSNYC
    GEOLIFE = GEOLIFE
    PORTO = PORTO
    MNISTSequential = MNISTSequential


class Models(CustomEnum):
    """Models."""
    DiffTraj = 'difftraj'
    UNetGAN = 'unetgan'
    UNetVAE = 'unetvae'
    PrivTrace = 'privtrace'


class NormOption(CustomEnum):
    """Enum for the different normalization options"""

    BATCH_1D = 'batch1d'
    BATCH = 'batch'
    LAYER = 'layer'
    NONE = 'none'
    DROPOUT = 'dropout'
    GROUP = 'group'
    INSTANCE = 'instance'
    WEIGHT = 'weight'


class ActivationOption(CustomEnum):
    """Enum for the different activation options"""

    RELU = 'relu'
    LEAKY_RELU = 'leaky_relu'
    TANH = 'tanh'
    SIGMOID = 'sigmoid'

class LRSchedulerType(CustomEnum):
    """Enum for the different LR Scheduler types"""

    COSINE = 'cosine'
    COSRES = 'cosres'
    MULTI_STEP = 'multi_step'
    NONE = 'none'


class VAEType(CustomEnum):
    BETA = 'Beta'
    BETAPLUS = 'Beta+'
    VANILLA = 'Vanilla'


class DatasetModes(CustomEnum):
    """Enumeration of dataset modes."""
    TRAIN = 'train'
    TEST = 'test'
    ALL = 'all'
    PATH = 'path'


class FeatureType(CustomEnum):
    CONTINUOUS = 'continuous'
    CATEGORICAL = 'categorical'

    def __str__(self) -> str:
        return self.value.upper()


class BetaSchedule(CustomEnum):
    """Enum for the different beta schedules for Diffusion Models"""
    LINEAR = 'linear'
    EXPONENTIAL = 'exponential'
    CONSTANT = 'constant'
    CYCLICAL = 'cyclical'

class JSDModes(CustomEnum):
    """Choice between Divergence and Distance"""
    DIVERGENCE = 'divergence'
    DISTANCE = 'distance'

################################################################
# Custom Enums for Conditional Information
################################################################

class ConditionalInfoSource(CustomEnum):
    """Enum for the different conditional information sources"""
    DIFFTRAJ = 'difftraj'  # Use the 8-dim conditional information from DiffTraj
    SAMPLE = 'sample'  # Use an entire sample as conditional information (needs embedding)

class NoiseType(CustomEnum):
    """Enum for the different noise types.
    The noise time also determines the norm that has to be used for bounding the vector norm.

    Laplace noise: L1 norm (Manhattan distance) ||x||_1 = sum(abs(x_i))
    Gaussian noise: L2 norm (Euclidean distance) ||x||_2 = sqrt(sum(x_i^2))
    """
    GAUSSIAN = 'gaussian'
    LAPLACE = 'laplace'
    VMF = 'vmf'

class ConditionalInfoClippingType(CustomEnum):
    """How to bound the norm of the conditional information."""
    NORM_CLIPPING = 'norm'  # Scale by multiplication with min(1, (C / ||v||_p))
    MAX_CLIPPING = 'max'  # Scale by multiplication with min(1, (C / max(abs(v_i))))

class ConditionalInfoEmbeddingType(CustomEnum):
    """
    How to embed the conditional information
    """
    PREGENERATED = 'pregenerated'  # Use pre-generated conditional information
    INTEGRATED_FC = 'integrated_fc'  # Use an integrated FC layer

class ConditionalInfoLocation(CustomEnum):
    """
    Where to add noise and/or clip in the Conditional Information Module
    """
    EMBEDDING = 'embedding'  # Add noise to the final embedding (end of embedding module)
    HIDDEN = 'hidden'  # Add noise to the hidden layers (middle of the embedding module)