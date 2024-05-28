# Dense-field lockbox results

Held-out procedural profiles: **1024**.
Intervals are nonparametric 95% hierarchical bootstrap intervals over
training seeds and profiles (10,000 draws).

| policy | information_mean | information_ci95_low | information_ci95_high | ratio_to_random | rmse_mean | path_length_mean |
| --- | --- | --- | --- | --- | --- | --- |
| greedy_information | 5.4627 | 5.4495 | 5.4763 | 2.0527 | 0.1627 | 54.0000 |
| rainbow | 5.3152 | 5.2910 | 5.3446 | 1.9973 | 0.1649 | 54.0332 |
| lawnmower | 3.4507 | 3.4265 | 3.4742 | 1.2967 | 0.2036 | 56.3633 |
| random | 2.6612 | 2.6325 | 2.6902 | 1.0000 | 0.2389 | 43.0039 |

I obtained these results with the procedural dense-field experiment. XBeach was not
used for this run; `docs/XBEACH.md` defines the optional adapter and scientific data
boundary.
