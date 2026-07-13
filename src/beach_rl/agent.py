"""Masked Rainbow DQN learner (Double Q, dueling C51, PER, n-step, NoisyNet)."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from beach_rl.config import AgentConfig, BeachConfig
from beach_rl.network import RainbowNetwork


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class RainbowAgent:
    def __init__(
        self,
        beach: BeachConfig,
        config: AgentConfig,
        channels: int,
        scalars: int,
        actions: int,
        device: str = "auto",
    ):
        self.beach_config = beach
        self.config = config
        self.actions = actions
        self.device = resolve_device(device)
        arguments = (
            channels,
            beach.height,
            beach.width,
            scalars,
            actions,
            config.atoms,
            config.hidden_dim,
            config.noisy_sigma,
        )
        self.online = RainbowNetwork(*arguments).to(self.device)
        self.target = RainbowNetwork(*arguments).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.train()
        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=config.learning_rate, eps=1e-5
        )
        self.support = torch.linspace(config.v_min, config.v_max, config.atoms, device=self.device)
        self.delta_z = (config.v_max - config.v_min) / (config.atoms - 1)
        self.updates = 0
        self.last_projection_clip_fraction = 0.0

    @staticmethod
    def _tensor(array: NDArray, device: torch.device, dtype: torch.dtype) -> Tensor:
        return torch.as_tensor(array, dtype=dtype, device=device)

    @torch.no_grad()
    def act(
        self,
        observation: dict[str, NDArray[np.float32]],
        action_mask: NDArray[np.bool_],
        deterministic: bool = False,
    ) -> int:
        previous_mode = self.online.training
        self.online.train(not deterministic)
        if not deterministic:
            self.online.reset_noise()
        spatial = self._tensor(observation["spatial"][None], self.device, torch.float32)
        scalars = self._tensor(observation["scalars"][None], self.device, torch.float32)
        q_values = self.online.q_values(spatial, scalars, self.support)[0]
        mask = self._tensor(action_mask, self.device, torch.bool)
        q_values = q_values.masked_fill(~mask, -torch.inf)
        action = int(q_values.argmax().item())
        self.online.train(previous_mode)
        return action

    def _project_distribution(
        self,
        next_probabilities: Tensor,
        rewards: Tensor,
        discounts: Tensor,
        dones: Tensor,
    ) -> Tensor:
        tz = rewards[:, None] + discounts[:, None] * (~dones)[:, None] * self.support[None]
        self.last_projection_clip_fraction = float(
            ((tz < self.config.v_min) | (tz > self.config.v_max)).float().mean().item()
        )
        tz = tz.clamp(self.config.v_min, self.config.v_max)
        b = (tz - self.config.v_min) / self.delta_z
        lower = b.floor().long()
        upper = b.ceil().long()
        projected = torch.zeros_like(next_probabilities)
        equal = lower == upper
        lower_weight = upper.float() - b + equal.float()
        upper_weight = b - lower.float()
        projected.scatter_add_(1, lower, next_probabilities * lower_weight)
        projected.scatter_add_(1, upper, next_probabilities * upper_weight)
        return projected

    def learn(self, batch: dict[str, NDArray]) -> tuple[float, NDArray[np.float32]]:
        device = self.device
        spatial = self._tensor(batch["spatial"], device, torch.float32)
        scalars = self._tensor(batch["scalars"], device, torch.float32)
        actions = self._tensor(batch["actions"], device, torch.long)
        rewards = self._tensor(batch["rewards"], device, torch.float32)
        next_spatial = self._tensor(batch["next_spatial"], device, torch.float32)
        next_scalars = self._tensor(batch["next_scalars"], device, torch.float32)
        next_masks = self._tensor(batch["next_masks"], device, torch.bool)
        dones = self._tensor(batch["dones"], device, torch.bool)
        discounts = self._tensor(batch["discounts"], device, torch.float32)
        weights = self._tensor(batch["weights"], device, torch.float32)
        masks = self._tensor(batch["masks"], device, torch.bool)
        experts = self._tensor(batch["experts"], device, torch.bool)

        self.online.train()
        self.online.reset_noise()
        self.target.reset_noise()
        batch_indices = torch.arange(actions.shape[0], device=device)
        all_log_probabilities = self.online.log_distribution(spatial, scalars)
        log_probabilities = all_log_probabilities[batch_indices, actions]

        with torch.no_grad():
            online_next_q = self.online.q_values(next_spatial, next_scalars, self.support)
            # Terminal masks can legitimately be all false in other environments.
            safe_masks = next_masks | dones[:, None]
            masked_q = online_next_q.masked_fill(~safe_masks, -torch.inf)
            next_actions = masked_q.argmax(dim=1)
            next_probabilities = self.target.distribution(next_spatial, next_scalars)[
                batch_indices, next_actions
            ]
            target_distribution = self._project_distribution(
                next_probabilities, rewards, discounts, dones
            )

        per_item_loss = -(target_distribution * log_probabilities).sum(dim=1)
        if experts.any() and self.config.demonstration_loss_weight > 0:
            q_values = (all_log_probabilities.exp() * self.support).sum(dim=-1)
            margin = torch.full_like(q_values, self.config.demonstration_margin)
            margin[batch_indices, actions] = 0.0
            competing = (q_values + margin).masked_fill(~masks, -torch.inf).max(dim=1).values
            chosen = q_values[batch_indices, actions]
            supervised = (competing - chosen).clamp_min(0.0)
            per_item_loss = per_item_loss + (
                self.config.demonstration_loss_weight * experts.float() * supervised
            )
        loss = (weights * per_item_loss).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.config.gradient_clip)
        self.optimizer.step()
        self.updates += 1
        priorities = per_item_loss.detach().cpu().numpy().astype(np.float32) + 1e-6
        return float(loss.detach().item()), priorities

    def update_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def save(self, path: str | Path, *, metadata: dict[str, Any] | None = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.online.state_dict(),
                "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "updates": self.updates,
                "beach_config": asdict(self.beach_config),
                "agent_config": asdict(self.config),
                "metadata": metadata or {},
            },
            path,
        )

    def load(self, path: str | Path, *, load_optimizer: bool = False) -> dict[str, Any]:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        stored_beach = checkpoint.get("beach_config")
        stored_agent = checkpoint.get("agent_config")
        if stored_beach is not None and stored_beach != asdict(self.beach_config):
            raise ValueError("checkpoint beach configuration does not match the supplied YAML")
        if stored_agent is not None and stored_agent != asdict(self.config):
            raise ValueError("checkpoint agent configuration does not match the supplied YAML")
        self.online.load_state_dict(checkpoint["model"])
        self.target.load_state_dict(checkpoint.get("target", checkpoint["model"]))
        if load_optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.updates = int(checkpoint.get("updates", 0))
        return dict(checkpoint.get("metadata", {}))
