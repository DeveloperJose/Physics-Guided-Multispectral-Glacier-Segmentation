# Reproducibility Sweep Results - 2026-06-08

All 57 runs finished under MLflow experiment `reproducibility`.

Runtime: 22h 41m 57s active training; 23h 09m 58s wall-clock.

## Debris-Covered Ice

Primary metric: test Debris IoU.

| Variant | n | Mean IoU | Std | Min | Max | Mean P | Mean R |
|---|---:|---:|---:|---:|---:|---:|---:|
| 15_full_threshold_2p0 | 1 | 0.405 |  | 0.405 | 0.405 | 0.625 | 0.535 |
| 17_full_threshold_10p0 | 1 | 0.391 |  | 0.391 | 0.391 | 0.750 | 0.450 |
| 16_full_threshold_5p0 | 1 | 0.379 |  | 0.379 | 0.379 | 0.708 | 0.449 |
| 08_physics_velocity_no_spectral | 3 | 0.334 | 0.059 | 0.270 | 0.388 | 0.685 | 0.404 |
| 02_landsat_dem | 3 | 0.331 | 0.019 | 0.310 | 0.346 | 0.501 | 0.508 |
| 09_dissertation_full_no_loss | 3 | 0.310 | 0.048 | 0.266 | 0.361 | 0.726 | 0.350 |
| 10_dissertation_full | 3 | 0.251 | 0.097 | 0.142 | 0.325 | 0.802 | 0.278 |
| 05_static_physics | 3 | 0.246 | 0.180 | 0.039 | 0.368 | 0.398 | 0.567 |
| 04_landsat_dem_spectral_hsv | 3 | 0.214 | 0.125 | 0.081 | 0.329 | 0.501 | 0.345 |
| 06_velocity_channels | 3 | 0.202 | 0.037 | 0.160 | 0.227 | 0.616 | 0.323 |
| 19_dissertation_full_robust | 1 | 0.198 |  | 0.198 | 0.198 | 0.272 | 0.422 |
| 07_velocity_loss | 3 | 0.197 | 0.072 | 0.147 | 0.279 | 0.866 | 0.208 |
| 01_landsat_only | 3 | 0.182 | 0.052 | 0.135 | 0.237 | 0.523 | 0.235 |
| 03_landsat_spectral_hsv | 3 | 0.166 | 0.022 | 0.142 | 0.182 | 0.466 | 0.245 |
| 18_static_physics_robust | 1 | 0.165 |  | 0.165 | 0.165 | 0.237 | 0.354 |
| 11_flow_accumulation_only | 1 | 0.096 |  | 0.096 | 0.096 | 0.130 | 0.267 |
| 12_tpi_only | 1 | 0.081 |  | 0.081 | 0.081 | 0.647 | 0.085 |
| 13_roughness_only | 1 | 0.080 |  | 0.080 | 0.080 | 0.097 | 0.312 |
| 14_curvature_only | 1 | 0.046 |  | 0.046 | 0.046 | 0.461 | 0.048 |

Best individual run: `reproducibility_dci_15_full_threshold_2p0_seed42_desktop_20260608_103356` with IoU 0.405, precision 0.625, recall 0.535.

## Clean Ice

Primary metric: test Clean Ice IoU.

| Variant | n | Mean IoU | Std | Min | Max | Mean P | Mean R |
|---|---:|---:|---:|---:|---:|---:|---:|
| 02_landsat_dem | 3 | 0.691 | 0.006 | 0.683 | 0.695 | 0.786 | 0.852 |
| 05_static_physics | 3 | 0.675 | 0.014 | 0.659 | 0.687 | 0.749 | 0.874 |
| 09_dissertation_full_no_loss | 3 | 0.664 | 0.021 | 0.650 | 0.688 | 0.718 | 0.900 |
| 03_landsat_spectral_hsv | 3 | 0.653 | 0.025 | 0.636 | 0.682 | 0.709 | 0.894 |
| 01_landsat_only | 3 | 0.651 | 0.003 | 0.647 | 0.653 | 0.710 | 0.887 |
| 10_dissertation_full | 3 | 0.617 | 0.073 | 0.535 | 0.674 | 0.655 | 0.921 |

Best individual run: `reproducibility_ci_02_landsat_dem_seed42_desktop_20260608_152403` with IoU 0.695, precision 0.824, recall 0.816.
