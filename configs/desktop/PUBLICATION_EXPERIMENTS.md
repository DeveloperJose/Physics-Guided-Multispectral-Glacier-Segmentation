# Comprehensive Reproducibility Experiments

All runs log to the single MLflow experiment `reproducibility`.

Total configs: 57

## Design

- DCI core ablations: 10 variants x 3 seeds = 30 runs.
- DCI sensitivity and component checks: 9 seed-42 runs.
- CI comparison ablations: 6 variants x 3 seeds = 18 runs.
- Multiclass is deferred because this paper is centered on binary CI/DCI behavior.
- Early stopping patience is 15 epochs to favor breadth; rerun the best models later with patience 30 if needed.

## Runtime Estimate

- Expected active training time with patience 15: about 24-26 hours
- With 30 second pauses: about 24.5-26.5 hours wall-clock
- Conservative upper bound: about 30 hours if full-channel runs train longer than the first batch.
