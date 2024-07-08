from dataclasses import replace

import numpy as np

from beach_rl.config import BeachConfig
from beach_rl.env import Action, BeachSamplingEnv
from beach_rl.simulator import SyntheticBeachGenerator


def test_reset_and_transitions_are_seed_deterministic() -> None:
    config = BeachConfig(height=8, width=12, horizon=12, sample_budget=3)
    first = BeachSamplingEnv(config)
    second = BeachSamplingEnv(config)
    obs_a, _ = first.reset(seed=123)
    obs_b, _ = second.reset(seed=123)
    np.testing.assert_array_equal(obs_a["spatial"], obs_b["spatial"])
    np.testing.assert_array_equal(obs_a["scalars"], obs_b["scalars"])
    for _ in range(2):
        out_a = first.step(int(Action.SAMPLE))
        out_b = second.step(int(Action.SAMPLE))
        assert out_a[1] == out_b[1]
        assert out_a[2:4] == out_b[2:4]
        assert out_a[4]["position"] == out_b[4]["position"]
        assert out_a[4]["samples"] == out_b[4]["samples"]
        np.testing.assert_array_equal(out_a[0]["scalars"], out_b[0]["scalars"])


def test_sample_reward_is_exact_information_gain() -> None:
    env = BeachSamplingEnv(BeachConfig(height=8, width=12, horizon=5, sample_budget=2))
    env.reset(seed=7)
    expected = env.potential_information()[env.position]
    _, reward, _, _, info = env.step(int(Action.SAMPLE))
    assert np.isclose(info["step_information"], expected)
    assert np.isclose(reward, expected + info["shaping_reward"])
    assert env.cumulative_information > 0


def test_action_mask_prevents_unsafe_moves_and_exhausted_sampling() -> None:
    config = BeachConfig(height=8, width=12, horizon=6, sample_budget=1)
    env = BeachSamplingEnv(config)
    env.reset(seed=9)
    mask = env.action_mask()
    assert mask.shape == (6,)
    assert mask.any()
    env.step(int(Action.SAMPLE))
    assert not env.action_mask()[int(Action.SAMPLE)]


def test_wait_is_safe_on_isolated_profile() -> None:
    config = BeachConfig(height=8, width=12, horizon=3, sample_budget=1)
    profile = SyntheticBeachGenerator(config).generate(91)
    isolated = np.zeros(profile.shape, dtype=np.bool_)
    isolated[profile.start] = True
    profile = replace(profile, traversable=isolated)
    env = BeachSamplingEnv(config)
    env.reset(seed=91, profile=profile)
    env.step(int(Action.SAMPLE))
    mask = env.action_mask()
    assert mask[int(Action.WAIT)]
    assert mask.sum() == 1
    env.step(int(Action.WAIT))


def test_rare_background_information_matches_closed_form() -> None:
    config = BeachConfig(
        height=12,
        width=48,
        horizon=12,
        sample_budget=10,
        information_regime="rare_hotspot",
        background_loading=0.005,
        rare_hotspot_loading=0.30,
        rare_hotspot_width=0.025,
    )
    env = BeachSamplingEnv(config)
    env.reset(seed=1234)
    feature = env.profile.features[env._flat_index()]
    np.testing.assert_array_equal(feature, np.full(18, config.background_loading))
    for _ in range(config.sample_budget):
        env.step(int(Action.SAMPLE))
    signal_to_noise = (
        config.sample_budget
        * 18
        * config.background_loading**2
        * config.prior_variance
        / config.observation_noise**2
    )
    expected = 0.5 * np.log1p(signal_to_noise)
    assert np.isclose(env.cumulative_information, expected, atol=1e-10)
