from beach_rl.agent import RainbowAgent
from beach_rl.animation import policy_rollout
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
