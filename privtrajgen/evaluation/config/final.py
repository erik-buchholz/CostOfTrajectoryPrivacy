#!/usr/bin/env python3
""" """
from typing import Dict, Union

from privtrajgen.config import Datasets, ConditionalInfoSource, ConditionalInfoLocation, NoiseType, Models
from .privtrace import PrivTraceConfig
from .baseconfig import DPConfig, BaseConfig, ConditionalInfoConfig, UNetVAEConfig
from .difftraj_config import DiffTrajConfig
from .gan_config import UNetGANConfig, GANLossOptions
from ..split_dataset import get_train_set_len
from ...utils import compute_delta

###############################################################################
# Parameters
###############################################################################
STEPS = 100_000
SAVE_FREQ = STEPS // 10
PLOT_FREQ = STEPS // 100
NUM_RUNS = 1  # Number of runs for training - should always be one on HPC due to parallelisation
START_RUN_ID = 1  # Start ID for the runs
KFOLD = 5  # 1 for no kfold, x > 1 for x-fold cross-validation

# DP-SGD Batch Sizes (Model and dataset dependent)
# Points of Diminishing returns (eps=10.0, steps=20k,100k):
# - Geolife: 1,554 / 695  --> Use 2048 to be safe
# - Porto: 34,864 / 15,592  --> Go as high as possible
# The max physical batch sizes are 1024 for GAN, 1000 for VAE and 800 for DiffTraj. We don't want more than
# 2 virtual batches per physical batch.
# These are determined empirically
DP_BATCH_SIZES: Dict[Models, Dict[Datasets, int]] = {
    Models.DiffTraj: {
        Datasets.PORTO: 1600,
        Datasets.GEOLIFE: 1600
    },
    Models.UNetVAE: {
        Datasets.PORTO: 2000,
        Datasets.GEOLIFE: 2000
    },
    Models.UNetGAN: {
        Datasets.PORTO: 2048,
        Datasets.GEOLIFE: 2048
    }
}

final_evaluation: Dict[str, BaseConfig | PrivTraceConfig] = dict()
###############################################################################
# Naming Conventions
###############################################################################
# XXy: XX is the case ID, y \in {p, g, fs, fl} is the dataset
dataset_shorts = {"p": Datasets.PORTO, "g": Datasets.GEOLIFE}


# for porto, geolife, foursquare-small, and foursquare-large, respectively
def swap_datasets(config: BaseConfig, dataset: Datasets) -> BaseConfig:
    """Creat a new configuration only changing the dataset."""
    new_config = config.copy()
    new_config.dataset = dataset
    # Change batch size if DP-SGD is enabled
    if new_config.dp.dp:
        new_config.batch_size = DP_BATCH_SIZES[new_config.model_type][dataset]

    # GAN Consistency Checks --> We use different loss functions for DP and non-DP models
    # Make sure this is consistent
    if isinstance(new_config, UNetGANConfig):
        assert new_config.loss_option == GANLossOptions.WGAN_LP, "UNetGAN models should use WGAN-LP"

        if new_config.dp.dp:
            assert new_config.n_critic == 1, "DP-SGD models should use n_critic = 1"
        else:
            assert new_config.n_critic == 5, "Non-DP models should use n_critic = 5"
            # If we want the generator to make STEPS many steps, uncomment the following line
            # new_config.steps = STEPS * new_config.n_critic

    # Change delta for Gaussian CI
    if new_config.cond.noise == NoiseType.GAUSSIAN:
        new_config.cond.delta = compute_delta(get_train_set_len(dataset, n_splits=config.k_fold))
    return new_config


def add_config(config: BaseConfig, id: Union[str, int], only_porto: bool = False) -> None:
    """Add a configuration to the final evaluation dictionary."""
    # For the cases where config.cond.epsilon == 10.0, we attend a second case with id = idx and eps = 20.0
    _add_config(config, id, only_porto)
    if config.cond.epsilon == 10.0:
        new_config = config.copy()
        new_config.cond.epsilon = 20.0
        _add_config(new_config, f"{id}x", only_porto)


def _add_config(config: BaseConfig, id: Union[str, int], only_porto: bool = False) -> None:
    if not only_porto:
        for ds in dataset_shorts:
            final_evaluation[f"{id}{ds}"] = swap_datasets(config, dataset_shorts[ds])
    else:
        final_evaluation[f"{id}p"] = swap_datasets(config, Datasets.PORTO)


###############################################################################
# Sub-Configurations (Getters to create new configurations)
###############################################################################

################################
# DP-SGD Configurations
# - Delta is automatically chosen as 1 / n^1.1
################################

# (A) No DP-SGD
def no_dp() -> DPConfig:
    return DPConfig(dp=False)


# (B) DP-SGD
def dp_sgd(eps: float = 10.0) -> DPConfig:
    return DPConfig(dp=True, target_epsilon=eps)


################################
# Conditional Information Configurations
# Hyperparameter Tuning showed no influence of max_norm --> Fix at 1.0
max_norm_CI = 1.0
# The hidden dim had no huge impact - fix at 8 except for ablation study
hd_CI = 8
# Default Epsilon = 10.0 (to match DP-SGD)
eps_CI = 10.0
# Delta will be computed automatically
noise_schedule = True


# Configurations:
# 1. Unconditional Models --> No conditional information
# 2. Baseline CI w/o DP --> DiffTraj-type CI with no noise
# 3. Baseline CI w/ DP --> DiffTraj-type CI with Laplace noise
# 4. Novel CI w/o DP --> Sample-type CI with no noise [not evaluated]
# 5. Novel CI w/ DP --> Sample-type CI with Laplace noise
# There is no point for the novel method without DP (4) as it would be an autoencoder. Thus, no experiments are run.
# Above are the baseline configurations.
#
# Below are the ablation study configurations.
# 6. Novel CI w/ VMF Mechanism
# 7. Novel CI w/ Gaussian Mechanism
################################

# (2) DiffTraj-type conditional information w/ no DP
def dt_CI_no_dp() -> ConditionalInfoConfig:
    """Default CI."""
    return ConditionalInfoConfig(source=ConditionalInfoSource.DIFFTRAJ, dp=False, clip_norm=False)


# (3) DiffTraj-type conditional information w/ Laplace mechanism
def dt_CI_laplace() -> ConditionalInfoConfig:
    return ConditionalInfoConfig(source=ConditionalInfoSource.DIFFTRAJ, dp=True, clip_norm=True,
                                 location=ConditionalInfoLocation.HIDDEN, noise=NoiseType.LAPLACE,
                                 hidden_dim=hd_CI, epsilon=eps_CI, max_norm=max_norm_CI,
                                 noise_schedule=noise_schedule)


# (5) Sample-type conditional information w/ Laplace mechanism
def sample_CI_laplace(eps: float = eps_CI, hd: int = hd_CI) -> ConditionalInfoConfig:
    return ConditionalInfoConfig(source=ConditionalInfoSource.SAMPLE, dp=True, clip_norm=True,
                                 location=ConditionalInfoLocation.HIDDEN, noise=NoiseType.LAPLACE,
                                 hidden_dim=hd, epsilon=eps, max_norm=max_norm_CI,
                                 noise_schedule=noise_schedule)


# (6) Sample-type conditional information w/ VMF mechanism
def sample_CI_vmf(eps: float = eps_CI, hd: int = hd_CI) -> ConditionalInfoConfig:
    return ConditionalInfoConfig(source=ConditionalInfoSource.SAMPLE, dp=True, clip_norm=True,
                                 location=ConditionalInfoLocation.HIDDEN, noise=NoiseType.VMF,
                                 hidden_dim=hd, epsilon=eps, max_norm=max_norm_CI,
                                 noise_schedule=noise_schedule)


# (7) Sample-type conditional information w/ Gaussian mechanism
def sample_CI_gaussian(eps: float = eps_CI, hd: int = hd_CI) -> ConditionalInfoConfig:
    porto_delta = compute_delta(get_train_set_len(Datasets.PORTO, n_splits=KFOLD))
    return ConditionalInfoConfig(source=ConditionalInfoSource.SAMPLE, dp=True, clip_norm=True,
                                 location=ConditionalInfoLocation.HIDDEN, noise=NoiseType.GAUSSIAN,
                                 hidden_dim=hd, epsilon=eps, max_norm=max_norm_CI,
                                 noise_schedule=noise_schedule, delta=porto_delta)


###############################################################################
# DiffTraj
###############################################################################
# Unconditional (1A)
cfg: BaseConfig = DiffTrajConfig(dataset=Datasets.PORTO, num_runs=NUM_RUNS, k_fold=KFOLD, steps=STEPS,
                                 save_freq=SAVE_FREQ, plot_freq=PLOT_FREQ, conditional=False, dp=no_dp(),
                                 run_id=START_RUN_ID)
add_config(cfg, 1)
# Unconditional with DP-SGD (1B)
cfg = cfg.copy()
cfg.dp = dp_sgd()
add_config(cfg, 2)
# Conditional with DiffTraj-type CI and no DP-SGD (2A)
cfg = DiffTrajConfig(dataset=Datasets.PORTO, num_runs=NUM_RUNS, k_fold=KFOLD, steps=STEPS,
                     save_freq=SAVE_FREQ, plot_freq=PLOT_FREQ, conditional=True, dp=no_dp(),
                     cond=dt_CI_no_dp())
add_config(cfg, 3)
# Conditional with DiffTraj-type CI and DP-SGD (2B)
cfg = cfg.copy()
cfg.dp = dp_sgd()
add_config(cfg, 4)
# Conditional with DiffTraj-type CI and Laplace noise (3A)
cfg = cfg.copy()
cfg.cond = dt_CI_laplace()
cfg.dp = no_dp()
add_config(cfg, 5)
# Conditional with DiffTraj-type CI and Laplace noise and DP-SGD (3B)
cfg = cfg.copy()
cfg.dp = dp_sgd()
add_config(cfg, 6)
# Conditional with Sample-type CI and Laplace noise (5A)
cfg = cfg.copy()
cfg.cond = sample_CI_laplace()
cfg.dp = no_dp()
add_config(cfg, 7)
# Conditional with Sample-type CI and Laplace noise and DP-SGD (5B)
cfg = cfg.copy()
cfg.dp = dp_sgd()
add_config(cfg, 8)
###############################################################################
# UNetVAE
###############################################################################
# Unconditional w/o DP-SGD (1A)
cfg: BaseConfig = UNetVAEConfig(dataset=Datasets.PORTO, num_runs=NUM_RUNS, k_fold=KFOLD, steps=STEPS,
                                save_freq=SAVE_FREQ, plot_freq=PLOT_FREQ, dp=no_dp(), run_id=START_RUN_ID,
                                conditional=False)
add_config(cfg, 9)
# Unconditional with DP-SGD (1B)
cfg = cfg.copy()
cfg.dp = dp_sgd()
add_config(cfg, 10)
# Conditional with DiffTraj-type CI and no DP-SGD (2A)
cfg = UNetVAEConfig(dataset=Datasets.PORTO, num_runs=NUM_RUNS, k_fold=KFOLD, steps=STEPS,
                    save_freq=SAVE_FREQ, plot_freq=PLOT_FREQ, dp=no_dp(), run_id=START_RUN_ID,
                    conditional=True, cond=dt_CI_no_dp())
add_config(cfg, 11)
# Conditional with DiffTraj-type CI and DP-SGD (2B)
cfg = cfg.copy()
cfg.dp = dp_sgd()
add_config(cfg, 12)
# Conditional with DiffTraj-type CI and Laplace noise (3A)
cfg = cfg.copy()
cfg.cond = dt_CI_laplace()
cfg.dp = no_dp()
add_config(cfg, 13)
# Conditional with DiffTraj-type CI and Laplace noise and DP-SGD (3B)
cfg = cfg.copy()
cfg.dp = dp_sgd()
add_config(cfg, 14)
# Conditional with Sample-type CI and Laplace noise (5A)
cfg = cfg.copy()
cfg.cond = sample_CI_laplace()
cfg.dp = no_dp()
add_config(cfg, 15)
# Conditional with Sample-type CI and Laplace noise and DP-SGD (5B)
cfg = cfg.copy()
cfg.dp = dp_sgd()
add_config(cfg, 16)

###############################################################################
# UNetGAN
# Non-DP model use ~~standard GAN loss~~ WGAN-LP with n_critic = 5
# DP models use WGAN-LP.
###############################################################################
# Unconditional w/o DP-SGD (1A)
gan_cfg: UNetGANConfig = UNetGANConfig(dataset=Datasets.PORTO, num_runs=NUM_RUNS, k_fold=KFOLD, steps=STEPS,
                                       save_freq=SAVE_FREQ, plot_freq=PLOT_FREQ, dp=no_dp(), run_id=START_RUN_ID,
                                       conditional=False, loss_option=GANLossOptions.WGAN_LP, n_critic=5)
add_config(gan_cfg, 17)
# Unconditional with DP-SGD (1B)
gan_cfg = gan_cfg.copy()
gan_cfg.dp = dp_sgd()
gan_cfg.n_critic = 1
add_config(gan_cfg, 18)
# Conditional with DiffTraj-type CI and no DP-SGD (2A)
gan_cfg: UNetGANConfig = UNetGANConfig(dataset=Datasets.PORTO, num_runs=NUM_RUNS, k_fold=KFOLD, steps=STEPS,
                                       save_freq=SAVE_FREQ, plot_freq=PLOT_FREQ, dp=no_dp(), run_id=START_RUN_ID,
                                       conditional=True, cond=dt_CI_no_dp(), loss_option=GANLossOptions.WGAN_LP,
                                       n_critic=5)
add_config(gan_cfg, 19)
# Conditional with DiffTraj-type CI and DP-SGD (2B)
gan_cfg = gan_cfg.copy()
gan_cfg.dp = dp_sgd()
gan_cfg.loss_option = GANLossOptions.WGAN_LP
gan_cfg.n_critic = 1
add_config(gan_cfg, 20)
# Conditional with DiffTraj-type CI and Laplace noise (3A)
gan_cfg = gan_cfg.copy()
gan_cfg.cond = dt_CI_laplace()
gan_cfg.dp = no_dp()
gan_cfg.loss_option = GANLossOptions.WGAN_LP
gan_cfg.n_critic = 5
add_config(gan_cfg, 21)
# Conditional with DiffTraj-type CI and Laplace noise and DP-SGD (3B)
gan_cfg = gan_cfg.copy()
gan_cfg.dp = dp_sgd()
gan_cfg.loss_option = GANLossOptions.WGAN_LP
gan_cfg.n_critic = 1
add_config(gan_cfg, 22)
# Conditional with Sample-type CI and Laplace noise (5A)
gan_cfg = gan_cfg.copy()
gan_cfg.cond = sample_CI_laplace()
gan_cfg.dp = no_dp()
gan_cfg.loss_option = GANLossOptions.WGAN_LP
gan_cfg.n_critic = 5
add_config(gan_cfg, 23)
# Conditional with Sample-type CI and Laplace noise and DP-SGD (5B)
gan_cfg = gan_cfg.copy()
gan_cfg.dp = dp_sgd()
gan_cfg.loss_option = GANLossOptions.WGAN_LP
gan_cfg.n_critic = 1
add_config(gan_cfg, 24)

###############################################################################
# DP-UNetGAN Debugging (18)
###############################################################################
DEBUG_STEPS = 20_000
DEBUG_SF = 20_000
DEBUG_PF = 1_000

# debug_cfg = UNetGANConfig(dataset=Datasets.PORTO, num_runs=NUM_RUNS, k_fold=KFOLD, steps=DEBUG_STEPS,
#                           save_freq=DEBUG_SF, plot_freq=DEBUG_PF, dp=no_dp(), run_id=START_RUN_ID,
#                           conditional=True, cond=dt_CI_no_dp())

###############################################################################
# Ablation I: Impact of Epsilon on DP-SGD
# Use unconditional model (1B) and use epsilon 5, ~10~, 20, 50
# Use Porto only
###############################################################################
# Unconditional with DP-SGD (1B) - Epsilon 5.0
cfg = DiffTrajConfig(dataset=Datasets.PORTO, num_runs=NUM_RUNS, k_fold=KFOLD, steps=STEPS,
                     save_freq=SAVE_FREQ, plot_freq=PLOT_FREQ, conditional=False, dp=dp_sgd(5.0),
                     run_id=START_RUN_ID)
add_config(cfg, 25, only_porto=False)
# Unconditional with DP-SGD (1B) - Epsilon 20.0
cfg = cfg.copy()
cfg.dp = dp_sgd(20.0)
add_config(cfg, 26, only_porto=False)
# Unconditional with DP-SGD (1B) - Epsilon 50.0
cfg = cfg.copy()
cfg.dp = dp_sgd(50.0)
add_config(cfg, 27, only_porto=False)

###############################################################################
# Ablation II: Impact of Epsilon on Conditional Information
# Use case 5A (Sample-type CI with Laplace noise) and use epsilon 5, ~10~, 20, 50, 100
# Use Porto only
###############################################################################
# Conditional with Sample-type CI and Laplace noise (5A) - Epsilon 5.0
cfg = DiffTrajConfig(dataset=Datasets.PORTO, num_runs=NUM_RUNS, k_fold=KFOLD, steps=STEPS,
                     save_freq=SAVE_FREQ, plot_freq=PLOT_FREQ, conditional=True, dp=no_dp(),
                     cond=sample_CI_laplace(5.0), run_id=START_RUN_ID)
add_config(cfg, 28, only_porto=False)
# Conditional with Sample-type CI and Laplace noise (5A) - Epsilon 20.0
cfg = cfg.copy()
cfg.cond = sample_CI_laplace(20.0)
add_config(cfg, 29, only_porto=False)
# Conditional with Sample-type CI and Laplace noise (5A) - Epsilon 50.0
cfg = cfg.copy()
cfg.cond = sample_CI_laplace(50.0)
add_config(cfg, 30, only_porto=False)
# Conditional with Sample-type CI and Laplace noise (5A) - Epsilon 100.0
cfg = cfg.copy()
cfg.cond = sample_CI_laplace(100.0)
add_config(cfg, 31, only_porto=False)

###############################################################################
# Ablation III: Impact of Hidden Dim & Mechanism on Conditional Information
# 5A, 6A, 7A with hidden dims 8, 4, 256 and Laplace - VMF - Gaussian
# Use Porto only
###############################################################################
# Conditional with Sample-type CI and Laplace noise (5A) - HD 4
cfg = DiffTrajConfig(dataset=Datasets.PORTO, num_runs=NUM_RUNS, k_fold=KFOLD, steps=STEPS,
                     save_freq=SAVE_FREQ, plot_freq=PLOT_FREQ, conditional=True, dp=no_dp(),
                     cond=sample_CI_laplace(hd=4), run_id=START_RUN_ID)
add_config(cfg, 32, only_porto=False)
# Conditional with Sample-type CI and Laplace noise (5A) - HD 256
cfg = cfg.copy()
cfg.cond = sample_CI_laplace(hd=256)
add_config(cfg, 33, only_porto=False)
# Conditional with Sample-type CI and VMF noise (6A) - HD 8
cfg = cfg.copy()
cfg.cond = sample_CI_vmf(hd=4)
add_config(cfg, 34, only_porto=False)
# Conditional with Sample-type CI and VMF noise (6A) - HD 4
cfg = cfg.copy()
cfg.cond = sample_CI_vmf(hd=8)
add_config(cfg, 35, only_porto=False)
# Conditional with Sample-type CI and VMF noise (6A) - HD 256
cfg = cfg.copy()
cfg.cond = sample_CI_vmf(hd=256)
add_config(cfg, 36, only_porto=False)
# Conditional with Sample-type CI and Gaussian noise (7A) - HD 8
cfg = cfg.copy()
cfg.cond = sample_CI_gaussian(hd=4)
add_config(cfg, 37, only_porto=False)
# Conditional with Sample-type CI and Gaussian noise (7A) - HD 4
cfg = cfg.copy()
cfg.cond = sample_CI_gaussian(hd=8)
add_config(cfg, 38, only_porto=False)
# Conditional with Sample-type CI and Gaussian noise (7A) - HD 256
cfg = cfg.copy()
cfg.cond = sample_CI_gaussian(hd=256)
add_config(cfg, 39, only_porto=False)
