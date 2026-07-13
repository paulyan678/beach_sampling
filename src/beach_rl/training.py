"""Deterministic training loop for the masked Rainbow agent."""

from __future__ import annotations

import json
import platform
import random
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from beach_rl.agent import RainbowAgent
from beach_rl.config import ExperimentConfig
from beach_rl.env import BeachSamplingEnv
from beach_rl.policies import GreedyInformationPolicy
from beach_rl.replay import NStepAccumulator, OneStep, PrioritisedReplay
from beach_rl.simulator import BeachProfile


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def _prefill_demonstrations(
    config: ExperimentConfig,
    seed: int,
    replay: PrioritisedReplay,
    profile_provider: Callable[[int], BeachProfile] | None = None,
) -> None:
    """Add fixed model-based expert trajectories for DQfD-style pretraining."""
    env = BeachSamplingEnv(config.beach, seed=seed)
    for episode in range(config.agent.demonstration_episodes):
        profile_seed = seed * 1_000_000 + 800_000 + episode
        profile = profile_provider(profile_seed) if profile_provider else None
        observation, info = env.reset(seed=profile_seed, profile=profile)
        policy = GreedyInformationPolicy()
        policy.reset(env)
        accumulator = NStepAccumulator(config.agent.n_step, config.agent.gamma)
        while env.steps < config.beach.horizon:
            mask = info["action_mask"].copy()
            action = policy.act(env, observation)
            next_observation, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated
            step = OneStep(
                spatial=observation["spatial"],
                scalars=observation["scalars"],
                mask=mask,
                action=action,
                reward=reward,
                next_spatial=next_observation["spatial"],
                next_scalars=next_observation["scalars"],
                next_mask=next_info["action_mask"].copy(),
                done=done,
                expert=True,
            )
            for transition in accumulator.append(step):
                replay.add(transition)
            observation, info = next_observation, next_info
            if done:
                break


def train_agent(
    config: ExperimentConfig,
    seed: int,
    output_dir: str | Path,
    profile_provider: Callable[[int], BeachProfile] | None = None,
) -> tuple[RainbowAgent, pd.DataFrame, Path]:
    seed_everything(seed, config.training.deterministic_torch)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    env = BeachSamplingEnv(config.beach, seed=seed)
    agent = RainbowAgent(
        config.beach,
        config.agent,
        env.n_channels,
        env.n_scalars,
        env.n_actions,
        config.training.device,
    )
    replay = PrioritisedReplay(
        config.agent.replay_capacity,
        env.observation_shape,
        env.n_scalars,
        env.n_actions,
        config.agent.per_alpha,
        seed,
    )
    _prefill_demonstrations(config, seed, replay, profile_provider)
    accumulator = NStepAccumulator(config.agent.n_step, config.agent.gamma)
    episode = 0
    profile_seed = seed * 1_000_000 + episode
    profile = profile_provider(profile_seed) if profile_provider else None
    observation, info = env.reset(seed=profile_seed, profile=profile)
    episode_reward = 0.0
    losses: list[float] = []
    clipping_rates: list[float] = []
    rows: list[dict[str, float | int]] = []
    started = time.perf_counter()

    for update in range(config.agent.pretrain_updates):
        if len(replay) < config.agent.batch_size:
            break
        batch = replay.sample(config.agent.batch_size, config.agent.per_beta_start)
        loss, priorities = agent.learn(batch)
        replay.update_priorities(batch["indices"], priorities)
        losses.append(loss)
        clipping_rates.append(agent.last_projection_clip_fraction)
        if (update + 1) % max(config.agent.target_update_frequency // 4, 1) == 0:
            agent.update_target()

    for step in range(1, config.training.total_steps + 1):
        mask = info["action_mask"].copy()
        if len(replay) < config.agent.min_replay_size:
            action = int(np.random.choice(np.flatnonzero(mask)))
        else:
            action = agent.act(observation, mask)
        next_observation, reward, terminated, truncated, next_info = env.step(action)
        done = terminated or truncated
        one_step = OneStep(
            spatial=observation["spatial"],
            scalars=observation["scalars"],
            mask=mask,
            action=action,
            reward=reward,
            next_spatial=next_observation["spatial"],
            next_scalars=next_observation["scalars"],
            next_mask=next_info["action_mask"].copy(),
            done=done,
        )
        for transition in accumulator.append(one_step):
            replay.add(transition)
        episode_reward += reward

        if (
            len(replay) >= max(config.agent.min_replay_size, config.agent.batch_size)
            and step % config.agent.update_frequency == 0
        ):
            progress = min(step / config.agent.per_beta_steps, 1.0)
            beta = config.agent.per_beta_start + progress * (1.0 - config.agent.per_beta_start)
            batch = replay.sample(config.agent.batch_size, beta)
            loss, priorities = agent.learn(batch)
            replay.update_priorities(batch["indices"], priorities)
            losses.append(loss)
            clipping_rates.append(agent.last_projection_clip_fraction)
        if step % config.agent.target_update_frequency == 0:
            agent.update_target()

        if done:
            rows.append(
                {
                    "global_step": step,
                    "episode": episode,
                    "profile_seed": profile_seed,
                    "return": episode_reward,
                    "information_gain": env.cumulative_information,
                    "samples": env.samples,
                    "path_length": env.path_length,
                    "mean_loss": float(np.mean(losses[-100:])) if losses else np.nan,
                    "target_clip_fraction": (
                        float(np.mean(clipping_rates[-100:])) if clipping_rates else np.nan
                    ),
                }
            )
            episode += 1
            profile_seed = seed * 1_000_000 + episode
            profile = profile_provider(profile_seed) if profile_provider else None
            observation, info = env.reset(seed=profile_seed, profile=profile)
            episode_reward = 0.0
        else:
            observation, info = next_observation, next_info

        if step % config.training.checkpoint_every == 0:
            agent.save(
                output / f"checkpoint_step_{step}.pt",
                metadata={"seed": seed, "step": step},
            )

    checkpoint = output / "checkpoint_final.pt"
    agent.save(
        checkpoint,
        metadata={"seed": seed, "step": config.training.total_steps},
    )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "training.csv", index=False)
    metadata = {
        "seed": seed,
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(agent.device),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "config": config.to_dict(),
        "demonstration_transitions": min(
            config.agent.demonstration_episodes * config.beach.horizon,
            config.agent.replay_capacity,
        ),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return agent, frame, checkpoint
