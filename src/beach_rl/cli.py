"""Command-line entry points for the autonomous beach-sampling research project."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from beach_rl.agent import RainbowAgent
from beach_rl.animation import render_policy_animation
from beach_rl.config import ExperimentConfig
from beach_rl.env import BeachSamplingEnv
from beach_rl.evaluation import EvaluationResult, evaluate_policies, summarise_evaluation
from beach_rl.plotting import (
    plot_benchmark,
    plot_coverage,
    plot_learning_curves,
    plot_trajectory,
)
from beach_rl.policies import (
    GreedyInformationPolicy,
    LawnmowerPolicy,
    RainbowPolicy,
    RandomPolicy,
    UniformTargetPolicy,
)
from beach_rl.sensitivity import (
    evaluate_background_loading_sensitivity,
    plot_loading_sensitivity,
)
from beach_rl.simulator import BeachProfile, XBeachExportAdapter
from beach_rl.training import train_agent


def _case_files(root: str | None, split: str) -> list[Path]:
    if root is None:
        return []
    directory = Path(root) / split
    if split == "test" and not directory.is_dir():
        directory = Path(root)
    files = sorted(directory.glob("*.npz")) if directory.is_dir() else []
    if not files:
        raise ValueError(f"no .npz XBeach exports found for '{split}' in {directory}")
    return files


def _case_provider(
    config: ExperimentConfig,
    files: list[Path],
    start_seed: int | None = None,
) -> Callable[[int], BeachProfile] | None:
    if not files:
        return None
    adapter = XBeachExportAdapter(config.beach)

    def provide(seed: int) -> BeachProfile:
        index = seed % len(files) if start_seed is None else seed - start_seed
        if not 0 <= index < len(files):
            raise ValueError("XBeach case bank does not cover the requested split/seed")
        return adapter.load(files[index], seed)

    return provide


def _write_case_manifest(output: Path, root: str | None) -> None:
    if root is None:
        return
    records = []
    for path in sorted(Path(root).rglob("*.npz")):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        records.append(
            {
                "path": str(path.relative_to(Path(root))),
                "sha256": digest.hexdigest(),
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    (output / "case_bank_manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


def _config_with_overrides(args: argparse.Namespace) -> ExperimentConfig:
    config = ExperimentConfig.from_yaml(args.config)
    training = config.training
    if getattr(args, "steps", None) is not None:
        training = replace(training, total_steps=args.steps)
    if getattr(args, "profiles", None) is not None:
        training = replace(training, eval_profiles=args.profiles)
    if getattr(args, "seeds", None):
        training = replace(training, seeds=tuple(args.seeds))
    if getattr(args, "eval_seed", None) is not None:
        training = replace(training, eval_seed=args.eval_seed)
    config = replace(config, training=training)
    config.validate()
    return config


def _new_agent(config: ExperimentConfig) -> RainbowAgent:
    env = BeachSamplingEnv(config.beach)
    return RainbowAgent(
        config.beach,
        config.agent,
        env.n_channels,
        env.n_scalars,
        env.n_actions,
        config.training.device,
    )


def _save_evaluation(result: EvaluationResult, output: Path) -> pd.DataFrame:
    output.mkdir(parents=True, exist_ok=True)
    result.episodes.to_csv(output / "episodes.csv", index=False)
    summary = summarise_evaluation(result.episodes)
    summary.to_csv(output / "summary.csv", index=False)
    np.savez_compressed(output / "coverage.npz", **result.coverage)
    plot_benchmark(summary, output / "benchmark.png")
    plot_coverage(result.coverage, output / "coverage.png")
    return summary


def _write_results_markdown(summary: pd.DataFrame, output: Path, config: ExperimentConfig) -> None:
    columns = [
        "policy",
        "information_mean",
        "information_ci95_low",
        "information_ci95_high",
        "ratio_to_random",
        "ratio_ci95_low",
        "ratio_ci95_high",
        "margin_over_100x_random",
        "margin_over_100x_ci95_low",
        "hundred_x_threshold_exceeded",
        "rmse_mean",
        "path_length_mean",
    ]
    table = summary[columns].copy()
    for column in [name for name in columns[1:] if name != "hundred_x_threshold_exceeded"]:
        table[column] = table[column].map(lambda value: f"{value:.4f}")
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    table_rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in table.itertuples(index=False, name=None)
    ]
    lines = [
        "# Experimental results",
        "",
        f"Held-out procedural profiles: **{config.training.eval_profiles}**. ",
        "Intervals are nonparametric 95% hierarchical bootstrap intervals over ",
        "training seeds and profiles (10,000 draws).",
        "",
        header,
        separator,
        *table_rows,
        "",
        "I obtained these results with the configured procedural beach experiment.",
        "XBeach was not used for this run; see `docs/XBEACH.md` for the adapter contract",
        "and the scientific boundary between coastal covariates and microplastic truth.",
        "",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def command_train(args: argparse.Namespace) -> None:
    config = _config_with_overrides(args)
    seed = args.seed if args.seed is not None else config.training.seeds[0]
    train_files = _case_files(args.xbeach_dir, "train") if args.xbeach_dir else []
    provider = _case_provider(config, train_files)
    _, frame, checkpoint = train_agent(config, seed, args.output, provider)
    _write_case_manifest(Path(args.output), args.xbeach_dir)
    print(json.dumps({"checkpoint": str(checkpoint), "episodes": len(frame)}, indent=2))


def command_evaluate(args: argparse.Namespace) -> None:
    config = _config_with_overrides(args)
    agent = _new_agent(config)
    agent.load(args.checkpoint)
    seeds = range(
        config.training.eval_seed,
        config.training.eval_seed + config.training.eval_profiles,
    )
    test_files = _case_files(args.xbeach_dir, "test") if args.xbeach_dir else []
    if test_files and len(test_files) < config.training.eval_profiles:
        raise ValueError("test case bank is smaller than the requested profile count")
    provider = _case_provider(config, test_files, config.training.eval_seed)
    factories = {
        "random": lambda seed: RandomPolicy(seed),
        "uniform_target": lambda seed: UniformTargetPolicy(seed),
        "lawnmower": lambda seed: LawnmowerPolicy(),
        "greedy_information": lambda seed: GreedyInformationPolicy(),
        "rainbow": lambda seed: RainbowPolicy(agent),
    }
    result = evaluate_policies(config.beach, factories, seeds, provider)
    _write_case_manifest(Path(args.output), args.xbeach_dir)
    summary = _save_evaluation(result, Path(args.output))
    print(summary.to_string(index=False))


def command_study(args: argparse.Namespace) -> None:
    config = _config_with_overrides(args)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8"
    )
    _write_case_manifest(output, args.xbeach_dir)

    train_files = _case_files(args.xbeach_dir, "train") if args.xbeach_dir else []
    validation_files = _case_files(args.xbeach_dir, "validation") if args.xbeach_dir else []
    test_files = _case_files(args.xbeach_dir, "test") if args.xbeach_dir else []
    if validation_files and len(validation_files) < config.training.validation_profiles:
        raise ValueError("validation case bank is smaller than the configured split")
    if test_files and len(test_files) < config.training.eval_profiles:
        raise ValueError("test case bank is smaller than the configured split")
    train_provider = _case_provider(config, train_files)
    validation_provider = _case_provider(config, validation_files, config.training.validation_seed)
    test_provider = _case_provider(config, test_files, config.training.eval_seed)

    training_frames: dict[int, pd.DataFrame] = {}
    agents: dict[int, RainbowAgent] = {}
    selection_rows: list[dict[str, int | float | str]] = []
    selection_seeds = range(
        config.training.validation_seed,
        config.training.validation_seed + config.training.validation_profiles,
    )
    for seed in config.training.seeds:
        run_directory = output / "training" / f"seed_{seed}"
        agent, frame, _ = train_agent(
            config,
            seed,
            run_directory,
            train_provider,
        )
        candidates = sorted(run_directory.glob("checkpoint_step_*.pt"))
        best_checkpoint: Path | None = None
        best_information = -np.inf
        for checkpoint in candidates:
            agent.load(checkpoint)
            candidate_result = evaluate_policies(
                config.beach,
                {"rainbow": lambda evaluation_seed, agent=agent: RainbowPolicy(agent)},
                selection_seeds,
                validation_provider,
            )
            mean_information = float(candidate_result.episodes["information_gain"].mean())
            selection_rows.append(
                {
                    "agent_seed": seed,
                    "checkpoint": checkpoint.name,
                    "validation_information_mean": mean_information,
                }
            )
            if mean_information > best_information:
                best_information = mean_information
                best_checkpoint = checkpoint
        if best_checkpoint is None:
            raise RuntimeError("training produced no model-selection checkpoints")
        agent.load(best_checkpoint)
        agents[seed] = agent
        training_frames[seed] = frame
    pd.DataFrame(selection_rows).to_csv(output / "model_selection.csv", index=False)
    plot_learning_curves(training_frames, output / "learning_curves.png")

    baseline_factories = {
        "random": lambda seed: RandomPolicy(seed),
        "uniform_target": lambda seed: UniformTargetPolicy(seed),
        "lawnmower": lambda seed: LawnmowerPolicy(),
        "greedy_information": lambda seed: GreedyInformationPolicy(),
    }

    def evaluate_split(
        profile_seeds: range,
        provider: Callable[[int], BeachProfile] | None,
    ) -> EvaluationResult:
        baseline = evaluate_policies(config.beach, baseline_factories, profile_seeds, provider)
        frames = [baseline.episodes]
        rainbow_coverages: list[np.ndarray] = []
        for agent_seed, agent in agents.items():
            evaluated = evaluate_policies(
                config.beach,
                {"rainbow": lambda evaluation_seed, agent=agent: RainbowPolicy(agent)},
                profile_seeds,
                provider,
            )
            evaluated.episodes["agent_seed"] = agent_seed
            frames.append(evaluated.episodes)
            rainbow_coverages.append(evaluated.coverage["rainbow"])
        coverage = dict(baseline.coverage)
        coverage["rainbow"] = np.mean(rainbow_coverages, axis=0)
        return EvaluationResult(pd.concat(frames, ignore_index=True), coverage)

    validation_seeds = range(
        config.training.validation_seed,
        config.training.validation_seed + config.training.validation_profiles,
    )
    validation = evaluate_split(validation_seeds, validation_provider)
    _save_evaluation(validation, output / "validation")

    profile_seeds = range(
        config.training.eval_seed,
        config.training.eval_seed + config.training.eval_profiles,
    )
    result = evaluate_split(profile_seeds, test_provider)
    summary = _save_evaluation(result, output)
    _write_results_markdown(summary, output, config)

    trajectory_env = BeachSamplingEnv(config.beach)
    trajectory_profile = (
        test_provider(config.training.eval_seed) if test_provider is not None else None
    )
    trajectory_env.reset(seed=config.training.eval_seed, profile=trajectory_profile)
    plot_trajectory(
        trajectory_env,
        RainbowPolicy(agents[config.training.seeds[0]]),
        output / "trajectory_rainbow.png",
    )
    print(summary.to_string(index=False))


def command_sensitivity(args: argparse.Namespace) -> None:
    config = _config_with_overrides(args)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    profile_seeds = range(
        config.training.eval_seed,
        config.training.eval_seed + config.training.eval_profiles,
    )
    frame = evaluate_background_loading_sensitivity(
        config.beach, args.background_loadings, profile_seeds
    )
    frame.to_csv(output / "background_loading_sensitivity.csv", index=False)
    plot_loading_sensitivity(frame, output / "background_loading_sensitivity.png")
    print(frame.to_string(index=False))


def command_animate(args: argparse.Namespace) -> None:
    config = ExperimentConfig.from_yaml(args.config)
    render_policy_animation(
        config,
        args.checkpoint,
        args.output,
        profile_seed=args.profile_seed,
        fps=args.fps,
    )
    print(json.dumps({"animation": str(args.output), "profile_seed": args.profile_seed}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="beach-rl",
        description="Bayesian deep-RL research for autonomous beach microplastic sampling",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--config", default="configs/research.yaml")
        subparser.add_argument("--output", required=True)
        subparser.add_argument("--steps", type=int, help="override training environment steps")
        subparser.add_argument("--profiles", type=int, help="override held-out profile count")
        subparser.add_argument("--eval-seed", type=int, help="override evaluation seed start")
        subparser.add_argument("--seeds", type=int, nargs="+", help="override training seeds")
        subparser.add_argument(
            "--xbeach-dir",
            help="case bank containing train/validation/test directories of .npz exports",
        )

    train = subparsers.add_parser("train", help="train one Rainbow agent")
    common(train)
    train.add_argument("--seed", type=int)
    train.set_defaults(function=command_train)

    evaluate = subparsers.add_parser("evaluate", help="evaluate one saved agent")
    common(evaluate)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.set_defaults(function=command_evaluate)

    study = subparsers.add_parser("study", help="run the complete multi-seed experiment")
    common(study)
    study.set_defaults(function=command_study)

    sensitivity = subparsers.add_parser(
        "sensitivity", help="sweep rare-regime background observation loading"
    )
    common(sensitivity)
    sensitivity.add_argument(
        "--background-loadings",
        type=float,
        nargs="+",
        default=[0.003, 0.004, 0.005, 0.006, 0.0075, 0.01],
    )
    sensitivity.set_defaults(function=command_sensitivity)

    animate = subparsers.add_parser(
        "animate", help="render a trained policy interacting with one simulated beach"
    )
    animate.add_argument("--config", default="configs/research.yaml")
    animate.add_argument("--checkpoint", required=True)
    animate.add_argument("--output", required=True)
    animate.add_argument("--profile-seed", type=int, default=50_000)
    animate.add_argument("--fps", type=int, default=8)
    animate.set_defaults(function=command_animate)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
