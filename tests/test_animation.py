import numpy as np

from beach_rl.agent import RainbowAgent
from beach_rl.animation import PolicyFrame, policy_rollout, sampling_mission_frames
from beach_rl.config import AgentConfig, BeachConfig, ExperimentConfig, TrainingConfig
from beach_rl.env import BeachSamplingEnv


def test_policy_rollout_records_initial_and_final_beliefs(tmp_path) -> None:
    beach = BeachConfig(height=8, width=12, horizon=4, sample_budget=2)
    agent_config = AgentConfig(
        atoms=11,
        hidden_dim=32,
        batch_size=2,
        replay_capacity=10,
        min_replay_size=2,
    )
    training = TrainingConfig(
        total_steps=1,
        seeds=(1,),
        eval_profiles=1,
        validation_profiles=1,
        checkpoint_every=1,
        device="cpu",
    )
    config = ExperimentConfig(beach=beach, agent=agent_config, training=training)
    env = BeachSamplingEnv(beach)
    agent = RainbowAgent(
        beach,
        agent_config,
        env.n_channels,
        env.n_scalars,
        env.n_actions,
        "cpu",
    )
    checkpoint = tmp_path / "agent.pt"
    agent.save(checkpoint)

    _, frames = policy_rollout(config, checkpoint, profile_seed=19)

    assert len(frames) == beach.horizon + 1
    assert frames[0].step == 0
    assert frames[-1].step == beach.horizon
    assert frames[-1].selected_action is None
    assert frames[-1].cumulative_information >= frames[0].cumulative_information


def _policy_frame(step: int, samples: int, action: int | None) -> PolicyFrame:
    return PolicyFrame(
        step=step,
        position=(0, step),
        trajectory=np.array([[0, step]], dtype=np.int64),
        sampled=np.empty((0, 2), dtype=np.int64),
        uncertainty=np.ones((2, 3), dtype=np.float64),
        action_values=np.arange(6, dtype=np.float32),
        selected_action=action,
        cumulative_information=float(samples),
        samples=samples,
    )


def test_sampling_mission_frames_end_at_first_complete_belief() -> None:
    full = [
        _policy_frame(0, 0, 3),
        _policy_frame(1, 1, 4),
        _policy_frame(2, 2, 0),
        _policy_frame(3, 2, None),
    ]
    presented = sampling_mission_frames(full, sample_budget=2)

    assert [item.step for item in presented] == [0, 1, 2]
    assert presented[-1].selected_action is None
    assert np.all(np.isnan(presented[-1].action_values))
    assert presented[-1].cumulative_information == full[2].cumulative_information


def test_sampling_mission_frames_keep_horizon_when_budget_is_not_reached() -> None:
    frames = [_policy_frame(0, 0, 3), _policy_frame(1, 1, None)]
    assert sampling_mission_frames(frames, sample_budget=99) is frames
