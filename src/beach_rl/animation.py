"""Animation of a trained policy acting in the Bayesian beach environment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from beach_rl.agent import RainbowAgent
from beach_rl.config import ExperimentConfig
from beach_rl.env import Action, BeachSamplingEnv


@dataclass(frozen=True)
class PolicyFrame:
    """One pre-decision belief state, including the network's masked values."""

    step: int
    position: tuple[int, int]
    trajectory: NDArray[np.int64]
    sampled: NDArray[np.int64]
    uncertainty: NDArray[np.float64]
    action_values: NDArray[np.float32]
    selected_action: int | None
    cumulative_information: float
    samples: int


def _frame(
    env: BeachSamplingEnv,
    values: NDArray[np.float32],
    selected_action: int | None,
) -> PolicyFrame:
    _, uncertainty = env.belief.predict(env.profile.features)
    sampled = np.argwhere(env.sample_counts > 0).astype(np.int64)
    return PolicyFrame(
        step=env.steps,
        position=env.position,
        trajectory=np.asarray(env.trajectory, dtype=np.int64),
        sampled=sampled,
        uncertainty=uncertainty.reshape(env.profile.shape),
        action_values=values.copy(),
        selected_action=selected_action,
        cumulative_information=env.cumulative_information,
        samples=env.samples,
    )


def policy_rollout(
    config: ExperimentConfig,
    checkpoint: str | Path,
    profile_seed: int,
) -> tuple[BeachSamplingEnv, list[PolicyFrame]]:
    """Run a deterministic trained policy and retain every belief state."""
    env = BeachSamplingEnv(config.beach, seed=profile_seed)
    observation, _ = env.reset(seed=profile_seed)
    agent = RainbowAgent(
        config.beach,
        config.agent,
        env.n_channels,
        env.n_scalars,
        env.n_actions,
        config.training.device,
    )
    agent.load(checkpoint)

    frames: list[PolicyFrame] = []
    while env.steps < env.config.horizon:
        values = agent.action_values(observation, env.action_mask(), deterministic=True)
        action = int(np.argmax(values))
        frames.append(_frame(env, values, action))
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    frames.append(
        _frame(
            env,
            np.full(env.n_actions, np.nan, dtype=np.float32),
            selected_action=None,
        )
    )
    return env, frames


def render_policy_animation(
    config: ExperimentConfig,
    checkpoint: str | Path,
    output: str | Path,
    profile_seed: int,
    fps: int = 8,
) -> None:
    """Render an accessible GIF showing motion, choices, and posterior learning."""
    # Import plotting lazily so rollouts and tests do not initialise a GUI or the
    # Matplotlib font parser when no visual output is requested.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    if fps <= 0:
        raise ValueError("fps must be positive")
    env, frames = policy_rollout(config, checkpoint, profile_seed)
    output_path = Path(output)
    if output_path.suffix.lower() != ".gif":
        raise ValueError("animation output must use the .gif extension")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    action_names = [action.name.lower() for action in Action]
    information = np.asarray([frame.cumulative_information for frame in frames])
    finite_values = np.concatenate(
        [frame.action_values[np.isfinite(frame.action_values)] for frame in frames[:-1]]
    )
    q_low = min(float(finite_values.min()), 0.0)
    q_high = max(float(finite_values.max()), 0.0)
    q_padding = max(0.1, 0.08 * (q_high - q_low))
    uncertainty_max = max(float(frame.uncertainty.max()) for frame in frames)

    figure = plt.figure(figsize=(10.4, 5.8))
    grid = figure.add_gridspec(3, 2, width_ratios=(1.35, 1.0), hspace=0.65, wspace=0.35)
    beach_axis = figure.add_subplot(grid[:, 0])
    uncertainty_axis = figure.add_subplot(grid[0, 1])
    value_axis = figure.add_subplot(grid[1, 1])
    information_axis = figure.add_subplot(grid[2, 1])
    beach_background = env.profile.deposition_risk
    traversability = np.where(env.profile.traversable, 1.0, np.nan)

    def draw(frame_index: int) -> None:
        frame = frames[frame_index]
        for axis in (beach_axis, uncertainty_axis, value_axis, information_axis):
            axis.clear()

        beach_axis.imshow(beach_background, origin="lower", cmap="viridis", vmin=0, vmax=1)
        beach_axis.imshow(traversability, origin="lower", cmap="gray", alpha=0.08)
        beach_axis.plot(
            frame.trajectory[:, 1],
            frame.trajectory[:, 0],
            color="white",
            linewidth=1.5,
            alpha=0.9,
            label="travel path",
        )
        if frame.sampled.size:
            beach_axis.scatter(
                frame.sampled[:, 1],
                frame.sampled[:, 0],
                c="#ff3b30",
                edgecolors="white",
                linewidths=0.5,
                s=34,
                label="sampled cell",
                zorder=4,
            )
        beach_axis.scatter(
            [frame.position[1]],
            [frame.position[0]],
            marker="*",
            c="#00ffff",
            edgecolors="black",
            linewidths=0.5,
            s=120,
            label="robot",
            zorder=5,
        )
        beach_axis.set(
            title="Simulated deposition prior and robot path",
            xlabel="cross-shore cell",
            ylabel="alongshore cell",
        )
        beach_axis.legend(frameon=False, loc="upper right", fontsize=8)

        uncertainty_image = uncertainty_axis.imshow(
            frame.uncertainty,
            origin="lower",
            cmap="magma",
            vmin=0,
            vmax=uncertainty_max,
            aspect="auto",
        )
        uncertainty_axis.scatter(
            [frame.position[1]], [frame.position[0]], marker="*", c="#00ffff", s=45
        )
        uncertainty_axis.set(
            title="Posterior uncertainty (latent SD)",
            xlabel="cross-shore cell",
            ylabel="alongshore cell",
        )
        if not hasattr(draw, "colorbar"):
            draw.colorbar = figure.colorbar(  # type: ignore[attr-defined]
                uncertainty_image, ax=uncertainty_axis, fraction=0.045, pad=0.03
            )
            draw.colorbar.set_label("standard deviation")  # type: ignore[attr-defined]

        valid = np.isfinite(frame.action_values)
        display_values = np.where(valid, frame.action_values, 0.0)
        colors = ["#9aa0a6"] * env.n_actions
        if frame.selected_action is not None:
            colors[frame.selected_action] = "#e45756"
        bars = value_axis.barh(action_names, display_values, color=colors)
        for bar, is_valid in zip(bars, valid, strict=True):
            if not is_valid:
                bar.set_alpha(0.18)
        value_axis.axvline(0, color="black", linewidth=0.7)
        value_axis.set_xlim(q_low - q_padding, q_high + q_padding)
        value_axis.invert_yaxis()
        value_axis.set(title="Masked action values", xlabel="expected discounted return Q")
        value_axis.grid(axis="x", alpha=0.2)

        information_axis.plot(
            np.arange(frame_index + 1),
            information[: frame_index + 1],
            color="#2a6fbb",
            linewidth=2,
        )
        information_axis.scatter(
            [frame_index], [information[frame_index]], color="#2a6fbb", s=22, zorder=3
        )
        information_axis.set_xlim(0, len(frames) - 1)
        information_axis.set_ylim(0, max(float(information.max()) * 1.08, 0.1))
        information_axis.set(
            title="Information collected over time",
            xlabel="environment step",
            ylabel="cumulative nats",
        )
        information_axis.grid(alpha=0.2)

        if frame.selected_action is None:
            decision = "episode complete"
        else:
            decision = f"next decision: {action_names[frame.selected_action]}"
        figure.suptitle(
            f"Step {frame.step}/{env.config.horizon} · {decision} · "
            f"samples {frame.samples}/{env.config.sample_budget} · "
            f"information {frame.cumulative_information:.3f} nats",
            fontsize=12,
        )

    held_frames = [0] * (2 * fps) + list(range(1, len(frames))) + [len(frames) - 1] * (2 * fps)
    animation = FuncAnimation(
        figure,
        draw,
        frames=held_frames,
        interval=1000 / fps,
        repeat=True,
    )
    try:
        animation.save(output_path, writer=PillowWriter(fps=fps), dpi=105)
    finally:
        plt.close(figure)
