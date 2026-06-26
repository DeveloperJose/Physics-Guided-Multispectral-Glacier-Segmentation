# Audit Prompt: GNSPI Fast Python vs Original IDL

## Task

Audit `scripts/run_hkh_gnspi_local_fast.py` against the original IDL `GNSPI.pro` for algorithmic fidelity. This is the **optimized** Python implementation using numba for speed. The goal is to find any remaining logic mismatches, not speed issues.

## Reference files (read all)

1. `analysis/gnspi_reference/gnspi_update_20130317/GNSPI update 20130317/GNSPI.pro` — original IDL code
2. `analysis/gnspi_reference/gnspi_update_20130317/GNSPI update 20130317/readme.txt` — IDL parameter docs
3. `analysis/gnspi_reference/gnspi_paper.txt` — RSE 2012 paper describing algorithm
4. `scripts/run_hkh_gnspi_local_fast.py` — Python implementation to audit
5. `scripts/run_hkh_nspi_local.py` — shared helpers (`load_stack`, `write_outputs`)

## Areas to compare

### 1. Inputs / masks
- IDL marks gap pixels as `fine1[i,j,*] eq 0` (any band zero) AND input not zero.
- Python uses `valid_optical` mask from band 7 of the stack.
- Check if mask changes the gap definition meaningfully.

### 2. K-means classification
- IDL uses `CLUST_WTS` + `CLUSTER` (ISODATA-like).
- Python uses sklearn `KMeans`.
- Different algorithms. Assess severity.

### 3. Class-wise OLS regression
- Verify `target ~ const + slope * input` formula matches IDL's `REGRESS`.
- Confirm both use same pixel selection (non-gap, same-class, ≥2 pixels).
- Check Python fallback when <2 pixels: IDL sets slope=1 const=0, what does Python do?

### 4. Residual image
- `dif = target - (input * slope + const)`
- Verified computed only over training pixels (non-gap, good class).

### 5. Semivariogram fitting **— CRITICAL**
- IDL: 1000 random non-gap pixels per class, `vgram2` (10 bins, binsize=4, range 0-40), exponential model via `CURVEFIT` with weights `count/γ²`. Falls back to no-nugget model if nugget < 0.
- Python `fit_exponential_variogram`: uses `sample_semiv=1000`, binned (10×4), `curve_fit` from scipy, fallback logic.
- Check: bin boundaries match? Curvfit weights match? Return parameter semantics match IDL's `result_semiv`?
- Key: Does Python's return `sill` mean the same thing as IDL's? (IDL stores A[2] = total_sill = psill+nugget as `result_semiv[0]`)

### 6. Similar-pixel selection
- RMSD formula: IDL uses hardcoded `total(...)/3.0` (test data had 3 bands). Python uses `/bands` (generalized).
- Threshold: IDL `mean(stddev(band)*2/class_num)`. Python same.
- Intersection across series images.
- Check: does Python's iteration in distance-sorted order (via pre-sorted `dys/dxs/d2s`) pick the same nearest pixels as IDL's `sort(dis_window2)`?

### 7. Sample selection fallback chain
- IDL: (1) ≥20 → kriging (2) 1-19 → kriging (3) 0 but same-class exist → mean of nearest tie-distance target pixels averaged with regression (4) no same-class → regression only.
- Python's `select_samples_chunk` uses modes 1/2/3/4.
- Verify each mode matches IDL. Especially mode 3: does Python compute `mean` of all tie-distance nearest pixels (like IDL's `where(dis_window2 eq min(dis_window2))` then `mean(sub_off[x,y])`)?

### 8. Ordinary kriging — check carefully
- **Covariance function**: IDL `Ch = sill*exp(-3D/range)` then `Ch[D=0] = sill+nugget`. Python `_cov_scalar` returns `sill+nugget` at h=0, `sill*exp(-3h/r)` otherwise. Does this match?
- **Parameter semantics**: IDL's `sill` in `O_kriging` = A[2] from fit = total_sill = psill + nugget. Python's `sill` from `fit_exponential_variogram` — what does it return?
- **Kriging matrix**: IDL uses `INVERT(Ch_plus) ## cs_plus`. Python uses `_solve_linear_inplace` on augmented system. Check matrix structure.
- **Kriging variance**: IDL `sill - cs_plus^T * C_plus^{-1} * cs_plus`. Python `kvar = sill - rhs_dot` where `rhs_dot = Σ(solution[j] * original_rhs[j])`. Both compute `σ² - c^T C^{-1} c`. Verify they produce the same value.

### 9. Uncertainty
- IDL: `uncertainty = error / fine0 * 100` where `error = 1.96 * σ_ok` (O_kriging returns result[1] already multiplied by 1.96).
- Python: `uncertainty = 196.0 * sqrt(kvar) / pred`. Does `196.0 = 1.96 * 100`? Check.

### 10. DN outlier replacement
- Both replace out-of-range predictions with mean of time-series values at that pixel.
- Check Python uses all series images including input (matches IDL's `mean(image_series[i,j,iband,*])` which includes the input image).

## Output format

Produce a table:

| Area | IDL behavior | Python behavior | Match? | Severity | Notes |
|------|-------------|----------------|--------|----------|-------|

Use these match categories:
- **faithful** — identical logic
- **acceptable adaptation** — intentionally different but equivalent or justified by data differences (HKH=6 bands vs test=3 bands)
- **real mismatch** — unintentional logic difference that would change results

## Important constraints

- Do not modify any files.
- Use `uv run python` if running scripts to inspect values.
- Focus on logic, not speed.
