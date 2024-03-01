"""Memory-efficient proportional prioritised replay with exact n-step returns."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Transition:
    spatial: NDArray[np.float32]
    scalars: NDArray[np.float32]
    mask: NDArray[np.bool_]
    action: int
    reward: float
    next_spatial: NDArray[np.float32]
    next_scalars: NDArray[np.float32]
    next_mask: NDArray[np.bool_]
    done: bool
    discount: float
    expert: bool = False


@dataclass(frozen=True)
class OneStep:
    spatial: NDArray[np.float32]
    scalars: NDArray[np.float32]
    mask: NDArray[np.bool_]
    action: int
    reward: float
    next_spatial: NDArray[np.float32]
    next_scalars: NDArray[np.float32]
    next_mask: NDArray[np.bool_]
    done: bool
    expert: bool = False


class NStepAccumulator:
    def __init__(self, n_step: int, gamma: float):
        self.n_step = n_step
        self.gamma = gamma
        self.buffer: deque[OneStep] = deque()

    def _make_transition(self) -> Transition:
        first = self.buffer[0]
        reward = 0.0
        final = first
        steps = 0
        for steps, item in enumerate(self.buffer, start=1):
            reward += self.gamma ** (steps - 1) * item.reward
            final = item
            if item.done or steps == self.n_step:
                break
        return Transition(
            spatial=first.spatial,
            scalars=first.scalars,
            mask=first.mask,
            action=first.action,
            reward=reward,
            next_spatial=final.next_spatial,
            next_scalars=final.next_scalars,
            next_mask=final.next_mask,
            done=final.done,
            discount=self.gamma**steps,
            expert=first.expert,
        )

    def append(self, step: OneStep) -> list[Transition]:
        self.buffer.append(step)
        output: list[Transition] = []
        if step.done:
            while self.buffer:
                output.append(self._make_transition())
                self.buffer.popleft()
        elif len(self.buffer) >= self.n_step:
            output.append(self._make_transition())
            self.buffer.popleft()
        return output

    def clear(self) -> None:
        self.buffer.clear()


class PrioritisedReplay:
    """Sum-tree PER. Observations are stored as float16 and restored as float32."""

    def __init__(
        self,
        capacity: int,
        spatial_shape: tuple[int, ...],
        scalar_size: int,
        actions: int,
        alpha: float,
        seed: int,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)
        tree_capacity = 1
        while tree_capacity < capacity:
            tree_capacity *= 2
        self.tree_capacity = tree_capacity
        self.tree = np.zeros(2 * tree_capacity, dtype=np.float64)
        self.min_tree = np.full(2 * tree_capacity, np.inf, dtype=np.float64)
        self.spatial = np.empty((capacity, *spatial_shape), dtype=np.float16)
        self.scalars = np.empty((capacity, scalar_size), dtype=np.float32)
        self.masks = np.empty((capacity, actions), dtype=np.bool_)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.next_spatial = np.empty((capacity, *spatial_shape), dtype=np.float16)
        self.next_scalars = np.empty((capacity, scalar_size), dtype=np.float32)
        self.next_masks = np.empty((capacity, actions), dtype=np.bool_)
        self.dones = np.empty(capacity, dtype=np.bool_)
        self.discounts = np.empty(capacity, dtype=np.float32)
        self.experts = np.empty(capacity, dtype=np.bool_)
        self.position = 0
        self.size = 0
        self.max_priority = 1.0

    def __len__(self) -> int:
        return self.size

    def _set_priority(self, index: int, priority: float) -> None:
        tree_index = self.tree_capacity + index
        change = priority - self.tree[tree_index]
        self.min_tree[tree_index] = priority
        while tree_index >= 1:
            self.tree[tree_index] += change
            if tree_index > 1:
                parent = tree_index // 2
                self.min_tree[parent] = min(
                    self.min_tree[2 * parent], self.min_tree[2 * parent + 1]
                )
            tree_index //= 2

    def add(self, transition: Transition) -> None:
        i = self.position
        self.spatial[i] = transition.spatial
        self.scalars[i] = transition.scalars
        self.masks[i] = transition.mask
        self.actions[i] = transition.action
        self.rewards[i] = transition.reward
        self.next_spatial[i] = transition.next_spatial
        self.next_scalars[i] = transition.next_scalars
        self.next_masks[i] = transition.next_mask
        self.dones[i] = transition.done
        self.discounts[i] = transition.discount
        self.experts[i] = transition.expert
        self._set_priority(i, self.max_priority**self.alpha)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _retrieve(self, mass: float) -> int:
        index = 1
        while index < self.tree_capacity:
            left = 2 * index
            if mass <= self.tree[left]:
                index = left
            else:
                mass -= self.tree[left]
                index = left + 1
        return min(index - self.tree_capacity, self.size - 1)

    def sample(self, batch_size: int, beta: float) -> dict[str, NDArray]:
        if self.size < batch_size:
            raise ValueError("not enough replay entries for a batch")
        total = self.tree[1]
        segment = total / batch_size
        masses = (np.arange(batch_size) + self.rng.random(batch_size)) * segment
        indices = np.array([self._retrieve(float(mass)) for mass in masses], dtype=np.int64)
        probabilities = self.tree[self.tree_capacity + indices] / total
        weights = (self.size * probabilities) ** (-beta)
        minimum_probability = self.min_tree[1] / total
        maximum_weight = (self.size * minimum_probability) ** (-beta)
        weights /= maximum_weight
        return {
            "indices": indices,
            "weights": weights.astype(np.float32),
            "spatial": self.spatial[indices].astype(np.float32),
            "scalars": self.scalars[indices],
            "masks": self.masks[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_spatial": self.next_spatial[indices].astype(np.float32),
            "next_scalars": self.next_scalars[indices],
            "next_masks": self.next_masks[indices],
            "dones": self.dones[indices],
            "discounts": self.discounts[indices],
            "experts": self.experts[indices],
        }

    def update_priorities(self, indices: NDArray[np.int64], priorities: NDArray) -> None:
        for index, priority in zip(indices, priorities, strict=True):
            value = max(float(priority), 1e-6)
            self.max_priority = max(self.max_priority, value)
            self._set_priority(int(index), value**self.alpha)
