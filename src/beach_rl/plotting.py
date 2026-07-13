"""Publication-oriented, deterministic plots for the benchmark."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from beach_rl.env import BeachSamplingEnv  # noqa: E402
from beach_rl.policies import Policy  # noqa: E402


def plot_learning_curves(training_frames: dict[int, pd.DataFrame], path: str | Path) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    for seed, frame in training_frames.items():
        if frame.empty:
            continue
        smooth = frame["information_gain"].rolling(20, min_periods=1).mean()
        axis.plot(frame["global_step"], smooth, alpha=0.75, label=f"seed {seed}")
    axis.set(xlabel="environment steps", ylabel="episode information gain (nats)")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_benchmark(summary: pd.DataFrame, path: str | Path) -> None:
    ordered = summary.sort_values("information_mean")
    error = np.vstack(
        [
            ordered["information_mean"] - ordered["information_ci95_low"],
            ordered["information_ci95_high"] - ordered["information_mean"],
        ]
    )
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.barh(ordered["policy"], ordered["information_mean"], xerr=error, capsize=3)
    axis.set(xlabel="cumulative mutual information (nats)")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_coverage(coverage: dict[str, np.ndarray], path: str | Path) -> None:
    names = list(coverage)
    figure, axes = plt.subplots(1, len(names), figsize=(4.0 * len(names), 3.4), squeeze=False)
    for index, (axis, name) in enumerate(zip(axes[0], names, strict=True)):
        image = axis.imshow(coverage[name], origin="lower", vmin=0, vmax=1, cmap="magma")
        axis.set_title(name)
        axis.set(xlabel="cross-shore cell", ylabel="alongshore cell" if index == 0 else "")
    figure.colorbar(image, ax=axes.ravel().tolist(), label="sampling frequency", shrink=0.8)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_trajectory(env: BeachSamplingEnv, policy: Policy, path: str | Path) -> None:
    profile = env.profile
    observation, _ = env.reset(seed=profile.seed, profile=profile)
    policy.reset(env)
    while env.steps < env.config.horizon:
        action = policy.act(env, observation)
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    trajectory = np.asarray(env.trajectory)
    figure, axis = plt.subplots(figsize=(7.2, 3.8))
    background = axis.imshow(env.profile.deposition_risk, origin="lower", cmap="viridis")
    axis.plot(trajectory[:, 1], trajectory[:, 0], color="white", linewidth=1.3, alpha=0.9)
    sampled = np.argwhere(env.sample_counts > 0)
    axis.scatter(sampled[:, 1], sampled[:, 0], c="red", s=22, label="sample")
    axis.scatter([trajectory[0, 1]], [trajectory[0, 0]], marker="*", c="cyan", s=70, label="start")
    axis.set(xlabel="cross-shore cell", ylabel="alongshore cell")
    axis.legend(frameon=False, loc="upper right")
    figure.colorbar(background, ax=axis, label="deposition prior")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
