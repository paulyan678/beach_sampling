"""Dueling categorical Q-network with factorised Gaussian NoisyNet layers."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class NoisyLinear(nn.Module):
    """Factorised NoisyNet layer from Fortunato et al. (2018)."""

    def __init__(self, in_features: int, out_features: int, sigma0: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        self.sigma0 = sigma0
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.weight_sigma, self.sigma0 / math.sqrt(self.in_features))
        nn.init.constant_(self.bias_sigma, self.sigma0 / math.sqrt(self.out_features))

    @staticmethod
    def _scaled_noise(size: int, *, device: torch.device) -> Tensor:
        noise = torch.randn(size, device=device)
        return noise.sign() * noise.abs().sqrt()

    def reset_noise(self) -> None:
        input_noise = self._scaled_noise(self.in_features, device=self.weight_mu.device)
        output_noise = self._scaled_noise(self.out_features, device=self.weight_mu.device)
        self.weight_epsilon.copy_(output_noise.outer(input_noise))
        self.bias_epsilon.copy_(output_noise)

    def forward(self, inputs: Tensor) -> Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight, bias = self.weight_mu, self.bias_mu
        return F.linear(inputs, weight, bias)


class RainbowNetwork(nn.Module):
    """CNN encoder plus dueling C51 distributional head."""

    def __init__(
        self,
        channels: int,
        height: int,
        width: int,
        scalars: int,
        actions: int,
        atoms: int,
        hidden_dim: int,
        sigma0: float,
    ):
        super().__init__()
        self.actions = actions
        self.atoms = atoms
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        encoded = 64 * ((height + 1) // 2) * ((width + 1) // 2) + scalars
        self.value_hidden = NoisyLinear(encoded, hidden_dim, sigma0)
        self.value_out = NoisyLinear(hidden_dim, atoms, sigma0)
        self.advantage_hidden = NoisyLinear(encoded, hidden_dim, sigma0)
        self.advantage_out = NoisyLinear(hidden_dim, actions * atoms, sigma0)

    def logits(self, spatial: Tensor, scalars: Tensor) -> Tensor:
        features = torch.cat((self.encoder(spatial), scalars), dim=1)
        value = self.value_out(F.relu(self.value_hidden(features))).view(-1, 1, self.atoms)
        advantage = self.advantage_out(F.relu(self.advantage_hidden(features))).view(
            -1, self.actions, self.atoms
        )
        return value + advantage - advantage.mean(dim=1, keepdim=True)

    def distribution(self, spatial: Tensor, scalars: Tensor) -> Tensor:
        return self.logits(spatial, scalars).softmax(dim=-1)

    def log_distribution(self, spatial: Tensor, scalars: Tensor) -> Tensor:
        return self.logits(spatial, scalars).log_softmax(dim=-1)

    def q_values(self, spatial: Tensor, scalars: Tensor, support: Tensor) -> Tensor:
        return (self.distribution(spatial, scalars) * support).sum(dim=-1)

    def reset_noise(self) -> None:
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()
