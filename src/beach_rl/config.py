"""Typed experiment configuration and validated YAML loading."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class BeachConfig:
    height: int = 10
    width: int = 18
    rbf_rows: int = 3
    rbf_cols: int = 6
    rbf_length_scale: float = 0.23
    horizon: int = 64
    sample_budget: int = 10
    observation_noise: float = 0.20
    prior_variance: float = 1.0
    movement_cost: float = 0.003
    invalid_action_cost: float = 0.05
    revisit_cost: float = 0.002
    wait_cost: float = 0.001
    potential_shaping_scale: float = 0.5
    shaping_gamma: float = 0.99

    def validate(self) -> None:
        if min(self.height, self.width, self.rbf_rows, self.rbf_cols) <= 1:
            raise ValueError("grid and RBF dimensions must exceed one")
        if not (0 < self.sample_budget <= self.horizon):
            raise ValueError("sample_budget must be in [1, horizon]")
        if self.observation_noise <= 0 or self.prior_variance <= 0:
            raise ValueError("Bayesian variances must be positive")
        if (
            self.potential_shaping_scale < 0
            or self.wait_cost < 0
            or not (0 <= self.shaping_gamma <= 1)
        ):
            raise ValueError("potential shaping scale/gamma are invalid")


@dataclass(frozen=True)
class AgentConfig:
    atoms: int = 51
    v_min: float = -0.5
    v_max: float = 8.0
    hidden_dim: int = 256
    gamma: float = 0.99
    n_step: int = 5
    learning_rate: float = 1.0e-4
    batch_size: int = 64
    replay_capacity: int = 40_000
    min_replay_size: int = 1_000
    update_frequency: int = 4
    target_update_frequency: int = 1_000
    gradient_clip: float = 10.0
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_steps: int = 50_000
    noisy_sigma: float = 0.5
    demonstration_episodes: int = 128
    pretrain_updates: int = 1_000
    demonstration_margin: float = 0.8
    demonstration_loss_weight: float = 1.0

    def validate(self) -> None:
        if self.atoms < 2 or self.v_min >= self.v_max:
            raise ValueError("invalid categorical value support")
        if not (0 <= self.gamma <= 1) or self.n_step < 1:
            raise ValueError("gamma must be in [0, 1] and n_step positive")
        if self.min_replay_size > self.replay_capacity:
            raise ValueError("min_replay_size cannot exceed replay_capacity")
        if self.demonstration_episodes < 0 or self.pretrain_updates < 0:
            raise ValueError("demonstration episodes/updates cannot be negative")
        if self.demonstration_margin < 0 or self.demonstration_loss_weight < 0:
            raise ValueError("demonstration margin/weight cannot be negative")


@dataclass(frozen=True)
class TrainingConfig:
    total_steps: int = 30_000
    seeds: tuple[int, ...] = (11, 29, 47)
    eval_profiles: int = 1_024
    eval_seed: int = 50_000
    validation_profiles: int = 128
    validation_seed: int = 20_000
    checkpoint_every: int = 10_000
    device: str = "auto"
    deterministic_torch: bool = True

    def validate(self) -> None:
        if (
            self.total_steps <= 0
            or self.eval_profiles <= 0
            or self.validation_profiles <= 0
            or not self.seeds
        ):
            raise ValueError("training steps, profiles, and seeds must be non-empty/positive")


@dataclass(frozen=True)
class ExperimentConfig:
    beach: BeachConfig = field(default_factory=BeachConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def validate(self) -> None:
        self.beach.validate()
        self.agent.validate()
        self.training.validate()
        if self.beach.potential_shaping_scale > 0 and not abs(
            self.beach.shaping_gamma - self.agent.gamma
        ) < 1e-12:
            raise ValueError("potential-shaping gamma must equal the agent gamma")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        allowed = {"beach", "agent", "training"}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"unknown configuration sections: {sorted(unknown)}")
        training = dict(raw.get("training", {}))
        if "seeds" in training:
            training["seeds"] = tuple(training["seeds"])
        config = cls(
            beach=BeachConfig(**raw.get("beach", {})),
            agent=AgentConfig(**raw.get("agent", {})),
            training=TrainingConfig(**training),
        )
        config.validate()
        return config
