"""Prespecified sensitivity analysis for the rare-wrackline stress test."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from beach_rl.config import BeachConfig  # noqa: E402
from beach_rl.evaluation import evaluate_policies  # noqa: E402
from beach_rl.policies import GreedyInformationPolicy, RandomPolicy  # noqa: E402


def evaluate_background_loading_sensitivity(
    config: BeachConfig,
    loadings: list[float],
    profile_seeds: range,
) -> pd.DataFrame:
    """Compare planner feasibility to primitive random for fixed loading values."""
    rows: list[dict[str, float | int]] = []
    for loading in loadings:
        candidate = replace(config, background_loading=loading)
        candidate.validate()
        result = evaluate_policies(
            candidate,
            {
                "random": lambda seed: RandomPolicy(seed),
                "greedy_information": lambda seed: GreedyInformationPolicy(),
            },
            profile_seeds,
        ).episodes
        means = result.groupby("policy")["information_gain"].mean()
        random_information = float(means["random"])
        planner_information = float(means["greedy_information"])
        rows.append(
            {
                "background_loading": loading,
                "profiles": result["profile_seed"].nunique(),
                "random_information_mean": random_information,
                "planner_information_mean": planner_information,
                "ratio_to_random": planner_information / random_information,
                "margin_over_100x_random": planner_information - 100.0 * random_information,
            }
        )
    return pd.DataFrame(rows)


def plot_loading_sensitivity(frame: pd.DataFrame, path: str | Path) -> None:
    figure, axis = plt.subplots(figsize=(6.6, 4.0))
    axis.plot(
        frame["background_loading"],
        frame["ratio_to_random"],
        marker="o",
        linewidth=1.8,
    )
    axis.axhline(100.0, color="black", linestyle="--", linewidth=1.0, label="100×")
    axis.set(
        xlabel="background observation loading",
        ylabel="greedy / primitive-random information ratio",
    )
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
