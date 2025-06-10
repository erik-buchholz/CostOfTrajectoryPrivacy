#!/usr/bin/env python3
"""PrivTrace Configuration."""
from typing import Dict, List, Union

from pydantic.dataclasses import dataclass
from simple_parsing import field

from privtrajgen.config import Datasets, Models
from privtrajgen.datasets import trajectory_lengths
from privtrajgen.evaluation.config import DPConfig
from privtrajgen.evaluation.config.baseconfig import Config, ValidationConfig, DataConfig


@dataclass(config=dict(validate_assignment=True))
class PrivTraceConfig(Config):
    dataset: Datasets = field(metadata={"custom_args": {"metavar": None}})
    epsilon: float = 10.0
    custom_K: bool = False  # Use custom K for the dataset
    custom_Kappa: bool = False  # Use custom Kappa for the dataset

    run_id: int = 0
    num_runs: int = 1
    seed: int = field(default=42)
    validation: ValidationConfig = field(default_factory=ValidationConfig)  # Validation Configuration

    @property
    def model_type(self) -> Models:
        return Models.PrivTrace

    @property
    def dp(self) -> DPConfig:
        """Differential Privacy Configuration."""
        return DPConfig(
            dp=True,
            target_epsilon=self.epsilon,
            target_delta=0,
            max_grad_norm=0.0,
        )

    @property
    def conditional(self) -> bool:
        """Indicates if the model is conditional."""
        return False

    @property
    def name(self) -> str:
        """Generate the name of the experiment."""
        name = f"PrivTrace_{self.dataset.upper()}_e{self.epsilon:.1f}_run{self.run_id:02d}"
        if self.custom_K:
            name += "custom-K"
        if self.custom_Kappa:
            name += "custom-Kappa"
        return name

    @property
    def data(self) -> DataConfig:
        """For compatibility with old DiffTraj configs"""
        return DataConfig(traj_length=trajectory_lengths[self.dataset], dataset=self.dataset)

    @property
    def k_fold(self) -> int:
        """Number of folds for k-fold cross-validation."""
        return 5


# Define PrivTrace Configurations
privTrace_configs: Dict[str, Union[PrivTraceConfig, List[PrivTraceConfig]]] = dict()
privTrace_configs_paper: Dict[str, Union[PrivTraceConfig, List[PrivTraceConfig]]] = dict()
epsilons = [10.0]
datasets = [Datasets.PORTO, Datasets.GEOLIFE]

for dataset in datasets:
    for epsilon in epsilons:
        name = f"privtrace_{dataset.name}_e{epsilon:.1f}"
        privTrace_configs_paper[name] = PrivTraceConfig(
            dataset=dataset,
            epsilon=epsilon,
        )
        name += "custom-K"
        privTrace_configs_paper[name] = PrivTraceConfig(
            dataset=dataset,
            epsilon=epsilon,
            custom_K=True,
        )

# Join the configs
privTrace_configs.update(privTrace_configs_paper)
# Add a test run with eps = 1000 and GEOLIFE
test_conf = PrivTraceConfig(dataset=Datasets.GEOLIFE, epsilon=1000.0)
privTrace_configs["test"] = test_conf

