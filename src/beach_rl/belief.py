"""Conjugate Bayesian linear-spatial belief used as the MDP information state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass
class GaussianSpatialBelief:
    """Posterior over RBF weights in ``y = phi(x)^T theta + epsilon``."""

    mean: FloatArray
    covariance: FloatArray
    noise_variance: float

    @classmethod
    def from_prior(
        cls, mean: FloatArray, prior_variance: float, noise_variance: float
    ) -> GaussianSpatialBelief:
        covariance = np.eye(mean.size, dtype=np.float64) * prior_variance
        return cls(mean.copy(), covariance, noise_variance)

    def information_gain(self, feature: FloatArray) -> float:
        """Exact conditional mutual information ``I(theta; y | D)`` in nats."""
        latent_variance = float(
            np.einsum("i,ij,j->", feature, self.covariance, feature, optimize=True)
        )
        return 0.5 * float(np.log1p(max(latent_variance, 0.0) / self.noise_variance))

    def update(self, feature: FloatArray, measurement: float) -> float:
        """Apply a stable rank-one Gaussian update and return information gain."""
        projected = np.einsum("ij,j->i", self.covariance, feature, optimize=True)
        innovation_variance = float(
            np.einsum("i,i->", feature, projected, optimize=True) + self.noise_variance
        )
        if innovation_variance <= 0 or not np.isfinite(innovation_variance):
            raise FloatingPointError("non-positive Bayesian innovation variance")
        gain = projected / innovation_variance
        innovation = measurement - float(
            np.einsum("i,i->", feature, self.mean, optimize=True)
        )
        info_gain = 0.5 * float(np.log(innovation_variance / self.noise_variance))
        self.mean = self.mean + gain * innovation
        self.covariance = self.covariance - np.outer(projected, projected) / innovation_variance
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        # Round-off can create tiny negative eigenvalues after many rank-one updates.
        eigenvalues, eigenvectors = np.linalg.eigh(self.covariance)
        if eigenvalues.min() < -1e-9:
            raise FloatingPointError("posterior covariance lost positive semidefiniteness")
        if eigenvalues.min() < 0:
            self.covariance = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
        return info_gain

    def predict(self, features: FloatArray) -> tuple[FloatArray, FloatArray]:
        mean = np.einsum("ij,j->i", features, self.mean, optimize=True)
        latent_variance = np.einsum("ij,jk,ik->i", features, self.covariance, features)
        return mean, np.sqrt(np.maximum(latent_variance, 0.0))

    def logdet(self) -> float:
        sign, value = np.linalg.slogdet(self.covariance)
        if sign <= 0:
            raise FloatingPointError("posterior covariance is not positive definite")
        return float(value)
