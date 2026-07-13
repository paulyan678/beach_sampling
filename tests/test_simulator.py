import numpy as np
import pytest

from beach_rl.config import BeachConfig
from beach_rl.simulator import SyntheticBeachGenerator, XBeachExportAdapter


def test_synthetic_profiles_are_reproducible_and_distinct() -> None:
    generator = SyntheticBeachGenerator(BeachConfig(height=8, width=12))
    first = generator.generate(22)
    repeat = generator.generate(22)
    other = generator.generate(23)
    np.testing.assert_array_equal(first.elevation, repeat.elevation)
    assert not np.allclose(first.elevation, other.elevation)
    assert first.traversable[first.start]


def test_xbeach_npz_adapter(tmp_path) -> None:
    y, x = np.mgrid[0:6, 0:9]
    path = tmp_path / "xbeach_export.npz"
    np.savez(
        path,
        zb=x / 4 - 0.5,
        zs=np.zeros_like(x, dtype=float),
        u=np.full_like(x, 0.2, dtype=float),
        v=np.zeros_like(x, dtype=float),
        sedero=np.maximum(x - 4, 0) / 10,
    )
    profile = XBeachExportAdapter(BeachConfig(height=8, width=12)).load(path, seed=2)
    assert profile.shape == (8, 12)
    assert profile.source.endswith("xbeach_export.npz")
    assert np.all((0 <= profile.deposition_risk) & (profile.deposition_risk <= 1))


def test_xbeach_adapter_requires_bed_elevation(tmp_path) -> None:
    path = tmp_path / "bad.npz"
    np.savez(path, u=np.ones((2, 2)))
    with pytest.raises(ValueError, match="zb"):
        XBeachExportAdapter(BeachConfig()).load(path, seed=1)


def test_xbeach_adapter_rejects_unregistered_shapes(tmp_path) -> None:
    path = tmp_path / "misregistered.npz"
    np.savez(path, zb=np.ones((4, 5)), u=np.ones((5, 5)))
    with pytest.raises(ValueError, match="registered"):
        XBeachExportAdapter(BeachConfig()).load(path, seed=1)
