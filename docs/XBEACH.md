# XBeach integration and scientific boundary

## XBeach is a coastal driver, not a microplastic fate model

XBeach solves coupled two-dimensional-horizontal wave, flow, mineral-sediment
transport, and bed-change processes. Its surfbeat mode resolves short-wave group
forcing, infragravity waves, unsteady currents, and swash. The official manual's
sediment formulations and density limits are intended for sand/gravel; the XBeach
`ccg` field is mineral sediment concentration and must not be presented as plastic
concentration.

This repository therefore uses XBeach only for elevation and hydrodynamic/morphology
covariates. A separate statistical microplastic layer maps those covariates to a
prior and noisy field. This separation is important for density, shape, settling,
biofouling, beaching, and backwashing effects absent from a sand transport law.

References: [official XBeach manual](https://xbeach.readthedocs.io/en/latest/xbeach_manual.html),
[Roelvink et al. 2009](https://doi.org/10.1016/j.coastaleng.2009.08.006), and the
[OpenEarth source mirror](https://github.com/openearth/xbeach).

## Stable export contract

`XBeachExportAdapter` reads a versioned NumPy `.npz` boundary rather than silently
depending on changing NetCDF dimension conventions.

| Key | Required | Interpretation |
| --- | --- | --- |
| `zb` | yes | bed elevation, positive upward |
| `zs` | no | survey-time water level |
| `u`, `v` | no | cell-centre velocity components |
| `sedero` | no | erosion/deposition proxy |
| `log_concentration` | no | externally modelled/measured microplastic truth |

All fields must be finite, two-dimensional, and registered on exactly the same
source shape and orientation. The adapter resamples
to the configured decision grid, computes water depth, speed/retention, deposition,
and a wet/dry-edge prior, and rejects a case with no cell shallower than the 0.05 m
traversability threshold or whose start component is smaller than the sample budget.
Missing optional hydrodynamic fields are explicit zeros. If `log_concentration` is
absent, the adapter labels contaminant truth as a synthetic prior draw; only an
external field earns the `exported_log_concentration` provenance label.

Example conversion after an XBeach NetCDF run (dimension/order checks belong in the
case-specific exporter):

```python
import numpy as np
import xarray as xr

dataset = xr.open_dataset("xboutput.nc")
np.savez_compressed(
    "case_0001.npz",
    zb=dataset["zb"].isel(globaltime=-1).values,
    zs=dataset["zs"].isel(globaltime=-1).values,
    u=dataset["u"].mean("globaltime").values,
    v=dataset["v"].mean("globaltime").values,
    sedero=(dataset["zb"].isel(globaltime=-1) - dataset["zb"].isel(globaltime=0)).values,
)
```

Do not silently transpose arrays. Verify XBeach's coastward x-axis, alongshore
y-axis, units, staggered-to-cell-centre interpolation, and time dimensions against
the exact executable/version. A 1-D transect is not a 2-D robot domain; use a 2DH
case or label an alongshore extension as synthetic.

## Recommended real case bank

For 1,000+ expensive XBeach cases, use cached surfbeat 2DH simulations and split by
whole case. Save canonical inputs, executable version/hash, `params.txt`, forcing,
stdout/stderr, return code, and output checksums. Sample beach/wave/tide parameters
with a space-filling design. Validate repeated-case determinism before relying on a
seed flag, and check finite outputs, time coverage, grid orientation, units,
`hh \approx max(zs-zb,0)`, and accessibility connectivity.

The current generated benchmark is correctly labeled `synthetic`; the machine used
for this reconstruction did not contain an XBeach executable or historical cases.

The CLI accepts an executable cached bank with non-overlapping directories:

```text
case-bank/
├── train/*.npz
├── validation/*.npz
└── test/*.npz
```

Run it with `beach-rl study ... --xbeach-dir case-bank`. Training cases may be
cycled across episodes; validation and test require at least the configured number
of distinct exports. The output stores a SHA-256 manifest of every input. A flat
directory is accepted only by the single-checkpoint `evaluate` command as a test
bank. Episode CSVs record both the profile path and contaminant-truth provenance.
