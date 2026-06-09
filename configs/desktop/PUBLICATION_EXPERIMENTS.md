# Threshold Follow-Up Experiments

All runs are DCI-only and log to the single MLflow experiment `reproducibility`.

Total configs: 31

## Rationale

The 2026-06-08 sweep showed the best DCI result from the full model with velocity loss threshold 2.0, but that threshold was only tested for seed 42. This batch tests whether that threshold advantage holds across seeds and whether the optimum is closer to 1.5, 2.0, or 2.5.

## Design

- Full DCI model with velocity loss thresholds 1.0, 1.5, 2.0, 2.5, 3.16, 5.0, and 10.0.
- Four seeds per threshold: 42, 1337, 2026, and 3407.
- Three seed-3407 controls: Landsat+DEM baseline, full model without velocity loss, and no-spectral physics+velocity at threshold 2.0.
- CI and multiclass are deferred; the clean-ice result was already stable enough and not the paper's main claim.

## Runtime Estimate

- 31 configs total.
- Expected active training time: about 14-16 hours.
- With 30 second pauses: about 14.5-16.5 hours wall-clock.
