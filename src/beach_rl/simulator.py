"""Synthetic coastal profiles and a narrow adapter for exported XBeach fields.

XBeach predicts hydrodynamics, sediment transport, and morphology; it does not
directly predict microplastic concentration.  This module therefore converts
coastal fields into a transparent deposition *prior* and keeps the contaminant
observation model separate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from beach_rl.config import BeachConfig

FloatArray = NDArray[np.float64]


def _normalise(array: FloatArray) -> FloatArray:
    low, high = float(np.nanmin(array)), float(np.nanmax(array))
    return (array - low) / max(high - low, 1e-12)


def rbf_design(config: BeachConfig, risk: FloatArray) -> FloatArray:
    """Return profile-conditioned radial basis features, shape ``(H*W, K)``."""
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, config.height),
        np.linspace(0.0, 1.0, config.width),
        indexing="ij",
    )
    centres_y, centres_x = np.meshgrid(
        np.linspace(0.0, 1.0, config.rbf_rows),
        np.linspace(0.0, 1.0, config.rbf_cols),
        indexing="ij",
    )
    distance2 = (
        (yy[..., None] - centres_y.ravel()) ** 2
        + (xx[..., None] - centres_x.ravel()) ** 2
    )
    base = np.exp(-0.5 * distance2 / config.rbf_length_scale**2)
    # Morphodynamics condition both the prior mean and where its uncertainty lies.
    profile_scale = 0.55 + 0.9 * risk[..., None]
    phi = base * profile_scale
    norms = np.sqrt(np.sum(phi**2, axis=(0, 1), keepdims=True))
    return (phi / np.maximum(norms, 1e-12)).reshape(config.height * config.width, -1)


def _ridge_projection(features: FloatArray, target: FloatArray) -> FloatArray:
    """Project a map onto the basis without platform-specific BLAS matmul warnings."""
    gram = np.einsum("ni,nj->ij", features, features, optimize=True)
    cross = np.einsum("ni,n->i", features, target, optimize=True)
    return np.linalg.solve(gram + 1e-3 * np.eye(features.shape[1]), cross)


@dataclass(frozen=True)
class BeachProfile:
    """One simulated beach and its Bayesian contaminant-model parameters."""

    elevation: FloatArray
    deposition_risk: FloatArray
    traversable: NDArray[np.bool_]
    features: FloatArray
    prior_mean: FloatArray
    true_weights: FloatArray
    start: tuple[int, int]
    seed: int
    source: str = "synthetic"
    truth_source: str = "prior_draw"

    @property
    def shape(self) -> tuple[int, int]:
        return self.elevation.shape

    def latent_log_concentration(self) -> FloatArray:
        values = np.einsum("ij,j->i", self.features, self.true_weights, optimize=True)
        return values.reshape(self.shape)


class SyntheticBeachGenerator:
    """Generate diverse, smooth beach profiles with an XBeach-like risk surrogate."""

    def __init__(self, config: BeachConfig):
        self.config = config

    def generate(self, seed: int) -> BeachProfile:
        rng = np.random.default_rng(seed)
        h, w = self.config.height, self.config.width
        y, x = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")

        phase = rng.uniform(0, 2 * np.pi)
        shoreline = 0.14 + rng.uniform(0.04, 0.13)
        shoreline = shoreline + rng.uniform(0.01, 0.05) * np.sin(2 * np.pi * y + phase)
        slope = rng.uniform(1.6, 2.8)
        berm_x = rng.uniform(0.38, 0.62)
        berm = rng.uniform(0.10, 0.35) * np.exp(-0.5 * ((x - berm_x) / 0.09) ** 2)
        alongshore = rng.uniform(0.03, 0.15) * np.sin(
            rng.integers(1, 4) * 2 * np.pi * y + rng.uniform(0, 2 * np.pi)
        )
        elevation = slope * (x - shoreline) + berm + alongshore

        # Proxy fields with direct physical interpretations: run-up/wrack deposition,
        # low-energy retention, erosion/deposition heterogeneity, and longshore forcing.
        wrackline = shoreline + rng.uniform(0.10, 0.24)
        runup_deposition = np.exp(-0.5 * ((x - wrackline) / rng.uniform(0.05, 0.11)) ** 2)
        low_energy = np.exp(-np.abs(np.gradient(elevation, axis=1)) / 0.30)
        morph_change = np.abs(np.gradient(np.gradient(elevation, axis=1), axis=1))
        longshore = 0.5 + 0.5 * np.sin(2 * np.pi * y + rng.uniform(0, 2 * np.pi))
        risk = _normalise(
            0.55 * runup_deposition
            + 0.20 * low_energy
            + 0.15 * morph_change
            + 0.10 * longshore
        )

        traversable = elevation > 0.02
        # Preserve a connected dry corridor on small/extreme randomly generated cases.
        traversable[:, max(1, int(0.42 * w)) :] = True
        desired = np.array([h // 2, max(1, int(0.45 * w))])
        locations = np.argwhere(traversable)
        start = tuple(locations[np.argmin(np.sum((locations - desired) ** 2, axis=1))])

        features = rbf_design(self.config, risk)
        target = -0.7 + 2.0 * risk.ravel()
        prior_mean = _ridge_projection(features, target)
        true_weights = prior_mean + rng.normal(
            0.0, np.sqrt(self.config.prior_variance), size=features.shape[1]
        )
        return BeachProfile(
            elevation=elevation.astype(np.float64),
            deposition_risk=risk.astype(np.float64),
            traversable=traversable,
            features=features,
            prior_mean=prior_mean,
            true_weights=true_weights,
            start=(int(start[0]), int(start[1])),
            seed=seed,
        )


class XBeachExportAdapter:
    """Build a profile from a stable ``.npz`` export of XBeach grid outputs.

    Required key: ``zb`` (bed elevation). Optional keys are ``zs`` (water level),
    ``u``, ``v``, and ``sedero`` (erosion/deposition). Arrays must be 2-D and are
    resampled to the configured research grid with deterministic nearest-neighbour
    indexing. This boundary avoids tying the core environment to one XBeach NetCDF
    version or variable naming convention.
    """

    def __init__(self, config: BeachConfig):
        self.config = config

    @staticmethod
    def _resize(array: FloatArray, shape: tuple[int, int]) -> FloatArray:
        rows = np.linspace(0, array.shape[0] - 1, shape[0]).round().astype(int)
        cols = np.linspace(0, array.shape[1] - 1, shape[1]).round().astype(int)
        return np.asarray(array[np.ix_(rows, cols)], dtype=np.float64)

    def load(self, path: str | Path, seed: int) -> BeachProfile:
        with np.load(path) as data:
            if "zb" not in data:
                raise ValueError("XBeach export must contain a 2-D 'zb' bed-elevation field")
            fields = {name: np.asarray(data[name], dtype=np.float64) for name in data.files}
        if any(value.ndim != 2 for value in fields.values()):
            raise ValueError("all exported XBeach fields must be two-dimensional")
        source_shapes = {value.shape for value in fields.values()}
        if len(source_shapes) != 1:
            raise ValueError("all exported XBeach fields must share one registered grid shape")
        if any(not np.all(np.isfinite(value)) for value in fields.values()):
            raise ValueError("exported XBeach fields must contain only finite values")

        shape = (self.config.height, self.config.width)
        zb = self._resize(fields["zb"], shape)
        zs = self._resize(fields.get("zs", np.zeros_like(fields["zb"])), shape)
        u = self._resize(fields.get("u", np.zeros_like(fields["zb"])), shape)
        v = self._resize(fields.get("v", np.zeros_like(fields["zb"])), shape)
        sedero = self._resize(fields.get("sedero", np.zeros_like(fields["zb"])), shape)
        water_depth = np.maximum(zs - zb, 0.0)
        speed = np.hypot(u, v)
        retention = 1.0 - _normalise(speed)
        deposition = _normalise(np.maximum(sedero, 0.0))
        wet_dry_edge = np.exp(-water_depth / max(np.nanstd(water_depth), 1e-6))
        risk = _normalise(0.45 * wet_dry_edge + 0.30 * retention + 0.25 * deposition)
        traversable = np.isfinite(zb) & (water_depth < 0.05)
        if not np.any(traversable):
            raise ValueError("XBeach export has no traversable cells at the 0.05 m threshold")

        locations = np.argwhere(traversable)
        desired = np.array([shape[0] // 2, shape[1] - 1])
        start_array = locations[np.argmin(np.sum((locations - desired) ** 2, axis=1))]
        start = (int(start_array[0]), int(start_array[1]))
        queue = [start]
        component = {start}
        while queue:
            row, col = queue.pop()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nxt = (row + dr, col + dc)
                if (
                    nxt not in component
                    and 0 <= nxt[0] < shape[0]
                    and 0 <= nxt[1] < shape[1]
                    and traversable[nxt]
                ):
                    component.add(nxt)
                    queue.append(nxt)
        if len(component) < self.config.sample_budget:
            raise ValueError("start component is too small for the configured sampling budget")
        features = rbf_design(self.config, risk)
        target = -0.7 + 2.0 * risk.ravel()
        prior_mean = _ridge_projection(features, target)
        if "log_concentration" in fields:
            measured_truth = self._resize(fields["log_concentration"], shape).ravel()
            true_weights = _ridge_projection(features, measured_truth)
            truth_source = "exported_log_concentration"
        else:
            rng = np.random.default_rng(seed)
            true_weights = prior_mean + rng.normal(
                0.0, np.sqrt(self.config.prior_variance), size=features.shape[1]
            )
            truth_source = "prior_draw"
        return BeachProfile(
            elevation=zb,
            deposition_risk=risk,
            traversable=traversable,
            features=features,
            prior_mean=prior_mean,
            true_weights=true_weights,
            start=start,
            seed=seed,
            source=str(Path(path)),
            truth_source=truth_source,
        )
