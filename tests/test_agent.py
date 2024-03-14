import torch

from beach_rl.agent import RainbowAgent
from beach_rl.config import AgentConfig, BeachConfig
from beach_rl.env import BeachSamplingEnv


def test_c51_projection_is_probability_distribution() -> None:
    beach = BeachConfig(height=8, width=12)
    config = AgentConfig(atoms=11, hidden_dim=32, batch_size=2, replay_capacity=10)
    env = BeachSamplingEnv(beach)
    agent = RainbowAgent(beach, config, env.n_channels, env.n_scalars, env.n_actions, "cpu")
    probabilities = torch.full((2, config.atoms), 1 / config.atoms)
    projected = agent._project_distribution(
        probabilities,
        torch.tensor([0.2, 1.0]),
        torch.tensor([0.99, 0.99]),
        torch.tensor([False, True]),
    )
    torch.testing.assert_close(projected.sum(dim=1), torch.ones(2))
    assert torch.all(projected >= 0)

    exact = torch.zeros((1, config.atoms))
    exact[0, 4] = 1.0
    exact_projection = agent._project_distribution(
        exact,
        torch.tensor([0.0]),
        torch.tensor([1.0]),
        torch.tensor([False]),
    )
    torch.testing.assert_close(exact_projection, exact)


def test_masked_action_selection_returns_valid_action() -> None:
    beach = BeachConfig(height=8, width=12)
    config = AgentConfig(atoms=11, hidden_dim=32, batch_size=2, replay_capacity=10)
    env = BeachSamplingEnv(beach)
    observation, info = env.reset(seed=4)
    agent = RainbowAgent(beach, config, env.n_channels, env.n_scalars, env.n_actions, "cpu")
    action = agent.act(observation, info["action_mask"], deterministic=True)
    assert info["action_mask"][action]
