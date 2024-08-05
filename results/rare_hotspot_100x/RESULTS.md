# Rare-wrackline lockbox results

Held-out procedural profiles: **2048**.
Intervals are nonparametric 95% hierarchical bootstrap intervals over
training seeds and profiles (10,000 draws).

| policy | information_mean | information_ci95_low | information_ci95_high | ratio_to_random | ratio_ci95_low | ratio_ci95_high | margin_over_100x_random | margin_over_100x_ci95_low | hundred_x_threshold_exceeded | rmse_mean | path_length_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| greedy_information | 5.8961 | 5.8961 | 5.8961 | 111.1111 | 110.9644 | 111.2742 | 0.5896 | 0.5826 | True | 0.0430 | 86.0000 |
| rainbow | 5.8563 | 5.8142 | 5.8880 | 110.3615 | 109.5782 | 110.9668 | 0.5498 | 0.5082 | True | 0.0431 | 86.0138 |
| uniform_target | 0.1464 | 0.1357 | 0.1572 | 2.7586 | 2.5583 | 2.9617 | -5.1601 | -5.1728 | False | 0.0555 | 90.5762 |
| random | 0.0531 | 0.0530 | 0.0531 | 1.0000 | 1.0000 | 1.0000 | -5.2534 | -5.2604 | False | 0.0556 | 68.4131 |
| lawnmower | 0.0297 | 0.0276 | 0.0320 | 0.5594 | 0.5205 | 0.6017 | -5.2768 | -5.2841 | False | 0.0558 | 91.3638 |

I obtained these results in the deliberately sparse synthetic information regime.
They are not calibrated XBeach or field-data results; `docs/RARE_HOTSPOT_100X.md`
defines the parameter sensitivity and interpretation limits.
