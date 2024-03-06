import numpy as np

from beach_rl.replay import NStepAccumulator, OneStep, PrioritisedReplay


def _step(reward: float, done: bool = False) -> OneStep:
    spatial = np.full((2, 3, 4), reward, dtype=np.float32)
    scalars = np.array([reward], dtype=np.float32)
    mask = np.ones(5, dtype=np.bool_)
    return OneStep(spatial, scalars, mask, 0, reward, spatial + 1, scalars + 1, mask, done)


def test_n_step_return_and_terminal_flush() -> None:
    accumulator = NStepAccumulator(3, 0.9)
    assert accumulator.append(_step(1.0)) == []
    assert accumulator.append(_step(2.0)) == []
    output = accumulator.append(_step(3.0, done=True))
    assert len(output) == 3
    assert np.isclose(output[0].reward, 1.0 + 0.9 * 2.0 + 0.9**2 * 3.0)
    assert np.isclose(output[1].reward, 2.0 + 0.9 * 3.0)
    assert all(item.done for item in output)


def test_prioritised_replay_shapes() -> None:
    accumulator = NStepAccumulator(1, 0.99)
    replay = PrioritisedReplay(8, (2, 3, 4), 1, 5, 0.6, seed=5)
    for value in range(6):
        replay.add(accumulator.append(_step(float(value)))[0])
    batch = replay.sample(4, beta=0.4)
    assert batch["spatial"].shape == (4, 2, 3, 4)
    assert batch["weights"].max() <= 1.0
    replay.update_priorities(batch["indices"], np.full(4, 2.0))
