import numpy as np
import pandas as pd

from beach_rl.config import BeachConfig
from beach_rl.evaluation import evaluate_policies, summarise_evaluation
from beach_rl.policies import GreedyInformationPolicy, RandomPolicy


def test_hundred_x_margin_requires_positive_paired_lower_bound() -> None:
    rows = []
    for profile_seed in range(20):
        rows.append(
            {
                "policy": "random",
                "profile_seed": profile_seed,
                "information_gain": 0.05,
                "return": 0.05,
                "samples": 10,
                "unique_samples": 8,
                "path_length": 50,
                "invalid_actions": 0,
                "posterior_rmse": 1.0,
            }
        )
        for agent_seed in (1, 2, 3):
            rows.append(
                {
                    "policy": "rainbow",
                    "profile_seed": profile_seed,
                    "agent_seed": agent_seed,
                    "information_gain": 6.0,
                    "return": 6.0,
                    "samples": 10,
                    "unique_samples": 10,
                    "path_length": 50,
                    "invalid_actions": 0,
                    "posterior_rmse": 0.1,
                }
            )
    summary = summarise_evaluation(pd.DataFrame(rows))
    rainbow = summary.set_index("policy").loc["rainbow"]
    assert np.isclose(rainbow["ratio_to_random"], 120.0)
    assert np.isclose(rainbow["margin_over_100x_random"], 1.0)
    assert bool(rainbow["hundred_x_threshold_exceeded"])


def test_rare_regime_is_information_theoretically_100x_feasible() -> None:
    config = BeachConfig(
        height=12,
        width=48,
        horizon=96,
        sample_budget=10,
        information_regime="rare_hotspot",
        rare_hotspot_loading=0.30,
        background_loading=0.005,
        rare_hotspot_width=0.025,
        shaping_gamma=1.0,
    )
    result = evaluate_policies(
        config,
        {
            "random": lambda seed: RandomPolicy(seed),
            "greedy_information": lambda seed: GreedyInformationPolicy(),
        },
        range(120_000, 120_032),
    )
    means = result.episodes.groupby("policy")["information_gain"].mean()
    assert means["greedy_information"] > 100.0 * means["random"]
