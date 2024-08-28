"""Paired held-out evaluation, bootstrap intervals, and randomisation tests."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from beach_rl.config import BeachConfig
from beach_rl.env import BeachSamplingEnv
from beach_rl.policies import Policy
from beach_rl.simulator import BeachProfile


@dataclass
class EvaluationResult:
    episodes: pd.DataFrame
    coverage: dict[str, np.ndarray]


def evaluate_policies(
    config: BeachConfig,
    policy_factories: dict[str, Callable[[int], Policy]],
    profile_seeds: Iterable[int],
    profile_provider: Callable[[int], BeachProfile] | None = None,
) -> EvaluationResult:
    rows: list[dict[str, float | int | str]] = []
    coverage = {
        name: np.zeros((config.height, config.width), dtype=np.float64) for name in policy_factories
    }
    seeds = list(profile_seeds)
    for profile_seed in seeds:
        for policy_index, (name, factory) in enumerate(policy_factories.items()):
            env = BeachSamplingEnv(config, seed=profile_seed)
            profile = profile_provider(profile_seed) if profile_provider else None
            observation, _ = env.reset(seed=profile_seed, profile=profile)
            policy = factory(profile_seed + 100_003 * policy_index)
            policy.reset(env)
            episode_return = 0.0
            invalid = 0
            while env.steps < config.horizon:
                action = policy.act(env, observation)
                observation, reward, terminated, truncated, info = env.step(action)
                episode_return += reward
                invalid += int(not info["valid_action"])
                if terminated or truncated:
                    break
            posterior_mean, _ = env.belief.predict(env.profile.features)
            truth = np.einsum(
                "ij,j->i", env.profile.features, env.profile.true_weights, optimize=True
            )
            rmse = float(np.sqrt(np.mean((posterior_mean - truth) ** 2)))
            unique_samples = int(np.count_nonzero(env.sample_counts))
            rows.append(
                {
                    "policy": name,
                    "profile_seed": profile_seed,
                    "return": episode_return,
                    "information_gain": env.cumulative_information,
                    "samples": env.samples,
                    "unique_samples": unique_samples,
                    "path_length": env.path_length,
                    "invalid_actions": invalid,
                    "posterior_rmse": rmse,
                    "profile_source": env.profile.source,
                    "truth_source": env.profile.truth_source,
                }
            )
            coverage[name] += (env.sample_counts > 0).astype(np.float64)
    for name in coverage:
        coverage[name] /= max(len(seeds), 1)
    return EvaluationResult(pd.DataFrame(rows), coverage)


def _bootstrap_mean_ci(
    values: np.ndarray, rng: np.random.Generator, draws: int = 10_000
) -> tuple[float, float]:
    n = values.size
    means = np.empty(draws, dtype=np.float64)
    chunk = 500
    for start in range(0, draws, chunk):
        count = min(chunk, draws - start)
        indices = rng.integers(0, n, size=(count, n))
        means[start : start + count] = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _bootstrap_policy_means(
    group: pd.DataFrame,
    rng: np.random.Generator,
    draws: int = 10_000,
) -> np.ndarray:
    """Resample both training seeds and held-out profiles when both are present."""
    has_agents = "agent_seed" in group and group["agent_seed"].notna().any()
    if has_agents:
        matrix = group.pivot_table(
            index="agent_seed", columns="profile_seed", values="information_gain", aggfunc="mean"
        ).to_numpy(dtype=np.float64)
        if np.isnan(matrix).any():
            raise ValueError("every evaluated agent seed must cover every profile")
    else:
        matrix = (
            group.groupby("profile_seed")["information_gain"]
            .mean()
            .to_numpy(dtype=np.float64)[None, :]
        )
    output = np.empty(draws, dtype=np.float64)
    chunk = 250
    for start in range(0, draws, chunk):
        count = min(chunk, draws - start)
        agent_indices = rng.integers(0, matrix.shape[0], size=(count, matrix.shape[0]))
        profile_indices = rng.integers(0, matrix.shape[1], size=(count, matrix.shape[1]))
        sampled = matrix[
            agent_indices[:, :, None],
            profile_indices[:, None, :],
        ]
        output[start : start + count] = sampled.mean(axis=(1, 2))
    return output


def _paired_bootstrap_comparison(
    group: pd.DataFrame,
    random_group: pd.DataFrame,
    rng: np.random.Generator,
    draws: int = 10_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    random_series = random_group.groupby("profile_seed")["information_gain"].mean()
    has_agents = "agent_seed" in group and group["agent_seed"].notna().any()
    if has_agents:
        policy_frame = group.pivot_table(
            index="agent_seed", columns="profile_seed", values="information_gain", aggfunc="mean"
        )
    else:
        policy_frame = group.groupby("profile_seed")["information_gain"].mean().to_frame().T
    common = policy_frame.columns.intersection(random_series.index)
    policy = policy_frame[common].to_numpy(dtype=np.float64)
    random = random_series.loc[common].to_numpy(dtype=np.float64)
    differences = np.empty(draws, dtype=np.float64)
    ratios = np.empty(draws, dtype=np.float64)
    hundred_x_margins = np.empty(draws, dtype=np.float64)
    chunk = 250
    for start in range(0, draws, chunk):
        count = min(chunk, draws - start)
        agent_indices = rng.integers(0, policy.shape[0], size=(count, policy.shape[0]))
        profile_indices = rng.integers(0, policy.shape[1], size=(count, policy.shape[1]))
        policy_mean = policy[
            agent_indices[:, :, None],
            profile_indices[:, None, :],
        ].mean(axis=(1, 2))
        random_mean = random[profile_indices].mean(axis=1)
        differences[start : start + count] = policy_mean - random_mean
        ratios[start : start + count] = policy_mean / np.maximum(random_mean, 1e-12)
        hundred_x_margins[start : start + count] = policy_mean - 100.0 * random_mean
    return differences, ratios, hundred_x_margins


def _paired_randomisation_p(
    differences: np.ndarray, rng: np.random.Generator, draws: int = 20_000
) -> float:
    observed = abs(float(differences.mean()))
    exceed = 0
    completed = 0
    chunk = 500
    for _ in range(0, draws, chunk):
        count = min(chunk, draws - completed)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(count, differences.size))
        exceed += int(np.sum(np.abs((signs * differences).mean(axis=1)) >= observed))
        completed += count
    return float((exceed + 1) / (draws + 1))


def summarise_evaluation(episodes: pd.DataFrame, seed: int = 2026) -> pd.DataFrame:
    """Summarise policies on paired profiles with nonparametric uncertainty."""
    rng = np.random.default_rng(seed)
    # If several trained agents share one policy name, average them within each profile
    # before resampling, preserving the profile as the independent evaluation unit.
    numeric = [
        "return",
        "information_gain",
        "samples",
        "unique_samples",
        "path_length",
        "invalid_actions",
        "posterior_rmse",
    ]
    per_profile = episodes.groupby(["policy", "profile_seed"], as_index=False)[numeric].mean()
    random_values = (
        per_profile[per_profile.policy == "random"]
        .set_index("profile_seed")["information_gain"]
        .sort_index()
    )
    rows: list[dict[str, float | int | str]] = []
    for policy, group in per_profile.groupby("policy", sort=True):
        values = group["information_gain"].to_numpy(dtype=np.float64)
        raw_group = episodes[episodes.policy == policy]
        bootstrap_means = _bootstrap_policy_means(raw_group, rng)
        ci_low, ci_high = np.quantile(bootstrap_means, [0.025, 0.975])
        row: dict[str, float | int | str] = {
            "policy": policy,
            "profiles": values.size,
            "information_mean": float(values.mean()),
            "information_median": float(np.median(values)),
            "information_std": float(values.std(ddof=1)),
            "information_ci95_low": ci_low,
            "information_ci95_high": ci_high,
            "return_mean": float(group["return"].mean()),
            "rmse_mean": float(group["posterior_rmse"].mean()),
            "unique_samples_mean": float(group["unique_samples"].mean()),
            "path_length_mean": float(group["path_length"].mean()),
            "invalid_actions_mean": float(group["invalid_actions"].mean()),
        }
        if not random_values.empty:
            paired = group.set_index("profile_seed")["information_gain"].sort_index()
            common = paired.index.intersection(random_values.index)
            difference = paired.loc[common].to_numpy() - random_values.loc[common].to_numpy()
            row["ratio_to_random"] = float(
                paired.loc[common].mean() / max(random_values.loc[common].mean(), 1e-12)
            )
            row["difference_to_random"] = float(difference.mean())
            row["log_ratio_to_random"] = float(np.log(max(row["ratio_to_random"], 1e-12)))
            row["margin_over_100x_random"] = float(
                paired.loc[common].mean() - 100.0 * random_values.loc[common].mean()
            )
            row["paired_randomisation_p"] = (
                1.0 if policy == "random" else _paired_randomisation_p(difference, rng)
            )
            random_group = episodes[episodes.policy == "random"]
            (
                bootstrap_difference,
                bootstrap_ratio,
                bootstrap_hundred_x_margin,
            ) = _paired_bootstrap_comparison(raw_group, random_group, rng)
            difference_low, difference_high = np.quantile(bootstrap_difference, [0.025, 0.975])
            ratio_low, ratio_high = np.quantile(bootstrap_ratio, [0.025, 0.975])
            margin_low, margin_high = np.quantile(bootstrap_hundred_x_margin, [0.025, 0.975])
            row["difference_ci95_low"] = float(difference_low)
            row["difference_ci95_high"] = float(difference_high)
            row["ratio_ci95_low"] = float(ratio_low)
            row["ratio_ci95_high"] = float(ratio_high)
            row["margin_over_100x_ci95_low"] = float(margin_low)
            row["margin_over_100x_ci95_high"] = float(margin_high)
            row["hundred_x_threshold_exceeded"] = bool(margin_low > 0.0)
        rows.append(row)
    return (
        pd.DataFrame(rows).sort_values("information_mean", ascending=False).reset_index(drop=True)
    )
