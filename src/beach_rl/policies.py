"""Deterministic and stochastic benchmark policies."""

from __future__ import annotations

from collections import deque
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from beach_rl.agent import RainbowAgent
from beach_rl.env import MOVES, Action, BeachSamplingEnv


class Policy(Protocol):
    def reset(self, env: BeachSamplingEnv) -> None: ...

    def act(
        self, env: BeachSamplingEnv, observation: dict[str, NDArray[np.float32]]
    ) -> int: ...


class RandomPolicy:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def reset(self, env: BeachSamplingEnv) -> None:
        del env

    def act(self, env: BeachSamplingEnv, observation: dict[str, NDArray[np.float32]]) -> int:
        del observation
        valid = np.flatnonzero(env.action_mask())
        return int(self.rng.choice(valid))


def _shortest_first_action(
    env: BeachSamplingEnv, target: tuple[int, int]
) -> int | None:
    """Breadth-first shortest path on the current profile's traversability graph."""
    if env.position == target:
        return int(Action.SAMPLE)
    queue = deque([(env.position, None)])
    visited = {env.position}
    while queue:
        position, first = queue.popleft()
        for action, (dr, dc) in MOVES.items():
            nxt = (position[0] + dr, position[1] + dc)
            if (
                nxt in visited
                or not (0 <= nxt[0] < env.config.height and 0 <= nxt[1] < env.config.width)
                or not env.profile.traversable[nxt]
            ):
                continue
            candidate_first = int(action) if first is None else first
            if nxt == target:
                return candidate_first
            visited.add(nxt)
            queue.append((nxt, candidate_first))
    return None


def _graph_distances(env: BeachSamplingEnv) -> dict[tuple[int, int], int]:
    queue = deque([env.position])
    distance = {env.position: 0}
    while queue:
        row, col = queue.popleft()
        for dr, dc in MOVES.values():
            nxt = (row + dr, col + dc)
            if (
                nxt not in distance
                and 0 <= nxt[0] < env.config.height
                and 0 <= nxt[1] < env.config.width
                and env.profile.traversable[nxt]
            ):
                distance[nxt] = distance[(row, col)] + 1
                queue.append(nxt)
    return distance


def _reachable(env: BeachSamplingEnv) -> list[tuple[int, int]]:
    return list(_graph_distances(env))


class LawnmowerPolicy:
    """Systematic coverage route with evenly spaced sample targets."""

    def __init__(self):
        self.targets: deque[tuple[int, int]] = deque()

    def reset(self, env: BeachSamplingEnv) -> None:
        reachable = set(_reachable(env))
        route: list[tuple[int, int]] = []
        for row in range(env.config.height):
            columns = (
                range(env.config.width)
                if row % 2 == 0
                else range(env.config.width - 1, -1, -1)
            )
            route.extend((row, col) for col in columns if (row, col) in reachable)
        if not route:
            route = [env.position]
        selected = np.linspace(0, len(route) - 1, env.config.sample_budget).round().astype(int)
        self.targets = deque(route[index] for index in selected)

    def act(self, env: BeachSamplingEnv, observation: dict[str, NDArray[np.float32]]) -> int:
        del observation
        while self.targets and env.sample_counts[self.targets[0]] > 0:
            self.targets.popleft()
        if not self.targets or env.samples >= env.config.sample_budget:
            valid_moves = np.flatnonzero(env.action_mask()[:4])
            return int(valid_moves[0]) if valid_moves.size else int(Action.WAIT)
        target = self.targets[0]
        action = _shortest_first_action(env, target)
        if action is None:
            self.targets.popleft()
            return self.act(env, {})
        return action


class GreedyInformationPolicy:
    """Path-aware one-step Bayesian-information heuristic.

    It chooses the reachable cell maximizing exact conditional mutual information
    divided by a sublinear travel penalty. This is a deliberately strong non-learning
    baseline for informative path planning.
    """

    def reset(self, env: BeachSamplingEnv) -> None:
        del env

    def act(self, env: BeachSamplingEnv, observation: dict[str, NDArray[np.float32]]) -> int:
        del observation
        if env.samples >= env.config.sample_budget:
            valid_moves = np.flatnonzero(env.action_mask()[:4])
            return int(valid_moves[0]) if valid_moves.size else int(Action.WAIT)
        information = env.potential_information()
        graph_distance = _graph_distances(env)
        cells = list(graph_distance)
        distance = np.array([graph_distance[cell] for cell in cells])
        gains = np.array([information[cell] for cell in cells])
        repeated = np.array([env.sample_counts[cell] > 0 for cell in cells])
        score = gains / (1.0 + distance) ** 0.35
        score[repeated] *= 0.7
        target = cells[int(np.argmax(score))]
        action = _shortest_first_action(env, target)
        return int(Action.SAMPLE) if action is None else action


class RainbowPolicy:
    def __init__(self, agent: RainbowAgent):
        self.agent = agent

    def reset(self, env: BeachSamplingEnv) -> None:
        del env

    def act(self, env: BeachSamplingEnv, observation: dict[str, NDArray[np.float32]]) -> int:
        return self.agent.act(observation, env.action_mask(), deterministic=True)
