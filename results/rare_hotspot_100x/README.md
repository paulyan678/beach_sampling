# Rare-wrackline sensitivity experiment

I use this directory to archive the lightweight outputs of the frozen
`configs/rare_hotspot_100x.yaml` sensitivity study. The directory name is retained
as the experiment identifier, not as a general research claim. Large PyTorch
checkpoints are excluded; raw episode rows, validation-only checkpoint scores,
training traces, metadata, figures, resolved configuration, and checksums remain.

Rainbow-DQfD obtained **5.8563 nats** versus primitive random's **0.0531 nats** on
2,048 lockbox profiles: **110.362×**, 95% hierarchical interval
**[109.578, 110.967]**. The paired 100× margin was 0.5498 nats [0.5082, 0.5817].

This is a synthetic nonstationary wrackline stress test, not a calibrated XBeach or
field-data result. The 110.362× number uses a 0.0531-nat primitive-random
denominator. Against the stronger uniform-waypoint baseline, the learned policy's
observed advantage is 5.710 nats (40.0×). I treat the dense-field approximately 2×
result as the main research finding.

See `docs/RARE_HOTSPOT_100X.md` for the feasibility bound, model, algorithm,
prespecified threshold diagnostic, sources, and interpretation limits.

`background_loading_sensitivity.csv` and `.png` record the validation-only sweep;
the planner/random ratio falls below 100× at loading 0.006 and above.
