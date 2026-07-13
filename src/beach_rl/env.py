"""Belief-state Markov decision process for autonomous beach sampling."""

from __future__ import annotations

from enum import IntEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from beach_rl.belief import GaussianSpatialBelief
from beach_rl.config import BeachConfig
from beach_rl.simulator import BeachProfile, SyntheticBeachGenerator


class Action(IntEnum):
    NORTH = 0
    SOUTH = 1
    WEST = 2
    EAST = 3
    SAMPLE = 4
    WAIT = 5


MOVES: dict[Action, tuple[int, int]] = {
    Action.NORTH: (-1, 0),
    Action.SOUTH: (1, 0),
    Action.WEST: (0, -1),
    Action.EAST: (0, 1),
}


class BeachSamplingEnv:
    """Small dependency-free environment with Gymnasium-compatible transition shape.

    The observation is a sufficient belief state: spatial posterior mean/standard
    deviation, morphology, visitation state, robot pose, and remaining budgets.
    Consequently the learning problem is an MDP even though the contaminant field
    itself is latent.
    """

    n_actions = len(Action)
    n_channels = 8
    def __init__(self, config: BeachConfig, seed: int = 0):
        self.config = config
        self.n_features = config.rbf_rows * config.rbf_cols
        # Budgets plus the complete Gaussian sufficient statistic (mean, covariance).
        self.n_scalars = 3 + self.n_features + self.n_features**2
        self.generator = SyntheticBeachGenerator(config)
        self._rng = np.random.default_rng(seed)
        self.profile: BeachProfile
        self.belief: GaussianSpatialBelief
        self.position = (0, 0)
        self.steps = 0
        self.samples = 0
        self.cumulative_information = 0.0
        self.path_length = 0
        self.visits = np.zeros((config.height, config.width), dtype=np.int32)
        self.sample_counts = np.zeros_like(self.visits)
        self.trajectory: list[tuple[int, int]] = []

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        return (self.n_channels, self.config.height, self.config.width)

    def reset(
        self, *, seed: int | None = None, profile: BeachProfile | None = None
    ) -> tuple[dict[str, NDArray[np.float32]], dict[str, Any]]:
        if seed is None:
            seed = int(self._rng.integers(0, 2**31 - 1))
        self._rng = np.random.default_rng(seed)
        self.profile = profile if profile is not None else self.generator.generate(seed)
        expected = (self.config.height, self.config.width)
        if self.profile.shape != expected:
            raise ValueError(f"profile shape {self.profile.shape} != configured shape {expected}")
        self.belief = GaussianSpatialBelief.from_prior(
            self.profile.prior_mean,
            self.config.prior_variance,
            self.config.observation_noise**2,
        )
        self.position = self.profile.start
        self.steps = self.samples = self.path_length = 0
        self.cumulative_information = 0.0
        self.visits = np.zeros(expected, dtype=np.int32)
        self.sample_counts = np.zeros(expected, dtype=np.int32)
        self.visits[self.position] = 1
        self.trajectory = [self.position]
        return self._observation(), self._info()

    def _flat_index(self, position: tuple[int, int] | None = None) -> int:
        row, col = self.position if position is None else position
        return row * self.config.width + col

    def action_mask(self) -> NDArray[np.bool_]:
        mask = np.ones(self.n_actions, dtype=np.bool_)
        row, col = self.position
        for action, (dr, dc) in MOVES.items():
            nr, nc = row + dr, col + dc
            mask[int(action)] = (
                0 <= nr < self.config.height
                and 0 <= nc < self.config.width
                and bool(self.profile.traversable[nr, nc])
            )
        mask[int(Action.SAMPLE)] = self.samples < self.config.sample_budget
        mask[int(Action.WAIT)] = True
        return mask

    def potential_information(self) -> NDArray[np.float64]:
        latent_variance = np.einsum(
            "ij,jk,ik->i",
            self.profile.features,
            self.belief.covariance,
            self.profile.features,
            optimize=True,
        )
        information = 0.5 * np.log1p(
            np.maximum(latent_variance, 0.0) / self.belief.noise_variance
        )
        return information.reshape(self.profile.shape)

    def _shaping_potential(self) -> float:
        """Path-aware state potential used for policy-invariant reward shaping."""
        information = self.potential_information()
        rows, columns = np.indices(self.profile.shape)
        distance = np.abs(rows - self.position[0]) + np.abs(columns - self.position[1])
        scores = information / (1.0 + distance) ** 0.35
        scores = np.where(self.profile.traversable, scores, -np.inf)
        return float(np.max(scores))

    def _observation(self) -> dict[str, NDArray[np.float32]]:
        mean, std = self.belief.predict(self.profile.features)
        h, w = self.profile.shape
        mean_map = mean.reshape(h, w)
        std_map = std.reshape(h, w)
        elevation = self.profile.elevation
        elevation = (elevation - elevation.mean()) / max(elevation.std(), 1e-6)
        robot = np.zeros((h, w), dtype=np.float64)
        robot[self.position] = 1.0
        channels = np.stack(
            [
                np.clip(elevation, -4, 4) / 4,
                self.profile.deposition_risk,
                np.tanh(mean_map / 3.0),
                std_map / max(np.sqrt(self.config.prior_variance), 1e-6),
                np.clip(self.sample_counts, 0, 3) / 3,
                robot,
                self.profile.traversable.astype(np.float64),
                np.clip(self.visits, 0, 5) / 5,
            ],
            axis=0,
        ).astype(np.float32)
        scalars = np.concatenate(
            (
                np.array(
                    [
                self.steps / self.config.horizon,
                self.samples / self.config.sample_budget,
                self.path_length / self.config.horizon,
                    ],
                    dtype=np.float64,
                ),
                self.belief.mean / 4.0,
                self.belief.covariance.ravel() / self.config.prior_variance,
            )
        ).astype(np.float32)
        return {"spatial": channels, "scalars": scalars}

    def _info(self) -> dict[str, Any]:
        return {
            "action_mask": self.action_mask(),
            "cumulative_information": self.cumulative_information,
            "samples": self.samples,
            "path_length": self.path_length,
            "position": self.position,
            "profile_seed": self.profile.seed,
        }

    def step(
        self, action: int
    ) -> tuple[dict[str, NDArray[np.float32]], float, bool, bool, dict[str, Any]]:
        if self.steps >= self.config.horizon:
            raise RuntimeError("step called after episode termination; call reset")
        try:
            selected = Action(action)
        except ValueError as exc:
            raise ValueError(f"action must be in [0, {self.n_actions - 1}]") from exc

        before_potential = self._shaping_potential()
        reward = 0.0
        valid = bool(self.action_mask()[int(selected)])
        sampled_information = 0.0
        if not valid:
            reward -= self.config.invalid_action_cost
        elif selected == Action.SAMPLE:
            index = self._flat_index()
            feature = self.profile.features[index]
            truth = float(
                np.einsum("i,i->", feature, self.profile.true_weights, optimize=True)
            )
            measurement = truth + float(self._rng.normal(0, self.config.observation_noise))
            sampled_information = self.belief.update(feature, measurement)
            self.sample_counts[self.position] += 1
            self.samples += 1
            self.cumulative_information += sampled_information
            reward += sampled_information
        elif selected == Action.WAIT:
            reward -= self.config.wait_cost
        else:
            dr, dc = MOVES[selected]
            self.position = (self.position[0] + dr, self.position[1] + dc)
            self.path_length += 1
            reward -= self.config.movement_cost
            if self.visits[self.position] > 0:
                reward -= self.config.revisit_cost

        self.steps += 1
        self.visits[self.position] += 1
        self.trajectory.append(self.position)
        terminated = self.steps >= self.config.horizon
        after_potential = 0.0 if terminated else self._shaping_potential()
        shaping_reward = self.config.potential_shaping_scale * (
            self.config.shaping_gamma * after_potential - before_potential
        )
        reward += shaping_reward
        info = self._info()
        info.update(
            {
                "valid_action": valid,
                "step_information": sampled_information,
                "shaping_reward": shaping_reward,
            }
        )
        return self._observation(), float(reward), terminated, False, info
