# LILA vs Bibek/Comprehensive Dataset Comparison

Generated 2026-06-20.

Compared processed datasets:
- LILA released conversion: `/home/devj/local-arch/data/HKH/lila_released_v1`
- Bibek-derived comprehensive dataset: `/home/devj/local-arch/data/HKH/comprehensive_v3`
- Bibek channel subsets: `comprehensive_v3_landsat*`, `comprehensive_v3_landsat_dem_flowacc_velmag`

## High-level differences

| Property | LILA released processed | Bibek comprehensive processed |
|---|---:|---:|
| Samples total | 548 | 1115 |
| Train / Val / Test | 383 / 55 / 110 | 804 / 138 / 173 |
| Channels | 15 | 24 full, subsets exist |
| Ignore label pixels | 0% | 28.82% |
| CI pixels, all pixels | 21.78% | 11.69% |
| DCI pixels, all pixels | 2.63% | 1.42% |
| CI pixels, valid-only | 21.78% | 16.42% |
| DCI pixels, valid-only | 2.63% | 1.99% |

## Split-level label density

### LILA released processed

| Split | n | with CI | with DCI | with both | mean valid label area | mean CI % valid | mean DCI % valid | median DCI % valid | p90 DCI % valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 383 | 383 | 327 | 327 | 100.00% | 21.29% | 2.77% | 1.64% | 7.08% |
| val | 55 | 55 | 45 | 45 | 100.00% | 22.94% | 2.13% | 1.51% | 4.34% |
| test | 110 | 110 | 96 | 96 | 100.00% | 22.90% | 2.38% | 1.54% | 6.93% |

### Bibek comprehensive processed

| Split | n | with CI | with DCI | with both | mean valid label area | mean CI % valid | mean DCI % valid | median DCI % valid | p90 DCI % valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 804 | 801 | 670 | 667 | 70.92% | 17.73% | 2.02% | 0.67% | 5.70% |
| val | 138 | 137 | 126 | 125 | 71.72% | 20.42% | 3.43% | 1.70% | 8.72% |
| test | 173 | 173 | 135 | 135 | 71.96% | 20.81% | 1.85% | 0.42% | 5.37% |

## Image/radiometry caveat

Both processed datasets store normalized arrays, but normalization differs:

- LILA released split arrays are already normalized/z-scored in the source split `.npy` files.
- Bibek/comprehensive `X.npy` arrays are also normalized, while `normalize_train.npy` stores raw-channel means/stds.
- Therefore direct `X.npy` value comparisons are not meaningful. Compare raw slices/tiles when evaluating coverage/artifacts.

## Raw image coverage checks

Using core Landsat reflective bands B1-B5+B7 and treating finite positive raw pixels as valid:

| Source | Unit | n | mean valid | median valid | p10 | p90 |
|---|---|---:|---:|---:|---:|---:|
| Bibek raw `/HKH_raw/Landsat7_2005` | 202 fishnet tiles | 202 | 82.48% | 93.60% | 52.55% | 97.80% |
| LILA all raw `slices/` | all generated 512 slices | 7095 | 49.24% | 46.13% | 0.00% | 100.00% |
| LILA selected released splits | selected 512 slices | 548 | 92.13% | 100.00% | 65.03% | 100.00% |

Interpretation:
- Released LILA split is much more selective: fewer slices, but selected slices have high raw reflective-band coverage.
- Bibek comprehensive has about 2× more samples and broader spatial/tile coverage, but also a large ignore-mask fraction (~29%) and lower positive class density.
- LILA has denser glacier labels per patch and especially more DCI fraction than Bibek comprehensive when measured over all pixels.

## Practical takeaway

- LILA is smaller but cleaner/denser as a benchmark split.
- Bibek comprehensive is larger and more complete as a broad HKH fishnet-derived training set, but includes many ignored/outside-label pixels and lower positive density.
- Comparing IoU across them is not apples-to-apples: split construction, valid-mask policy, channels, and imagery packaging differ.
