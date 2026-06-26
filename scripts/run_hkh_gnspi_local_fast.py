#!/usr/bin/env python3
"""Optimized local GNSPI experiment.

Faithful GNSPI structure, optimized implementation:
- global preprocess in numpy/sklearn
- chunked gap processing
- numba sample selection / similar-pixel intersection
- Python/numpy exact ordinary kriging per selected sample set and band

This avoids Python window scanning for every gap pixel while preserving GNSPI
selection and fallback logic.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import rasterio
from numba import njit, prange
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans

from run_hkh_nspi_local import load_stack, write_outputs

BAND_NAMES = ["B1", "B2", "B3", "B4", "B5", "B7"]


def _idl_variogram_with_nugget(h: np.ndarray, a0: float, a1: float, a2: float) -> np.ndarray:
    return a0 * np.exp(a1 * h) + a2


def _idl_variogram_no_nugget(h: np.ndarray, a0: float, a1: float) -> np.ndarray:
    return a0 * (1.0 - np.exp(a1 * h))


def fit_exponential_variogram(vals: np.ndarray, coords: np.ndarray) -> tuple[float, float, float]:
    """Fit GNSPI/IDL-style semivariogram.

    Returns (sill, nugget, range), matching IDL `result_semiv` semantics where
    `sill` is total sill used by `O_kriging`, not partial sill.
    """
    sample_semiv = 1000
    nbins = 10
    binsize = 4.0
    max_range = nbins * binsize
    n = len(vals)
    if n < sample_semiv:
        return 1.0, 0.0, 40.0

    rng = np.random.default_rng(0)
    idx = rng.choice(n, size=sample_semiv, replace=False)
    vals_s = vals[idx].astype(np.float64)
    coords_s = coords[idx].astype(np.float64)

    vargram = np.zeros(nbins, dtype=np.float64)
    vardist = np.zeros(nbins, dtype=np.float64)
    count = np.zeros(nbins, dtype=np.float64)
    for i in range(1, sample_semiv):
        dy = coords_s[i, 0] - coords_s[:i, 0]
        dx = coords_s[i, 1] - coords_s[:i, 1]
        d = np.sqrt(dy * dy + dx * dx)
        semi = 0.5 * (vals_s[i] - vals_s[:i]) ** 2
        valid = (d >= 0.0) & (d < max_range)
        if not np.any(valid):
            continue
        bin_idx = np.floor(d[valid] / binsize).astype(np.int64)
        for b, dist_v, semi_v in zip(bin_idx, d[valid], semi[valid]):
            vargram[b] += semi_v
            vardist[b] += dist_v
            count[b] += 1.0

    for b in range(nbins):
        if count[b] > 0:
            vardist[b] /= count[b]
            vargram[b] /= count[b]
        else:
            vardist[b] = 0.5 * (b * binsize + (b + 1) * binsize)

    valid_fit = (count > 0) & np.isfinite(vargram) & (vargram > 0)
    if valid_fit.sum() < 3:
        return 1.0, 0.0, 40.0

    x = vardist[valid_fit]
    y = vargram[valid_fit]
    weights = count[valid_fit] / np.maximum(y * y, 1e-12)
    sigma = 1.0 / np.sqrt(np.maximum(weights, 1e-12))
    max_var = max(float(np.max(y)), 1e-6)

    try:
        p0 = np.array([-max_var, -3.0 / max_range, max_var], dtype=np.float64)
        params, _ = curve_fit(
            _idl_variogram_with_nugget,
            x,
            y,
            p0=p0,
            sigma=sigma,
            absolute_sigma=False,
            maxfev=10000,
        )
        sill = float(params[2])
        nugget = float(params[2] + params[0])
        range_ = float(-3.0 / params[1]) if params[1] != 0 else 40.0
        if nugget >= 0.0 and sill > 0.0 and range_ > 0.0 and np.isfinite(range_):
            return sill, nugget, range_
    except (RuntimeError, ValueError, FloatingPointError):
        pass

    try:
        p0 = np.array([max_var, -3.0 / max_range], dtype=np.float64)
        params, _ = curve_fit(
            _idl_variogram_no_nugget,
            x,
            y,
            p0=p0,
            sigma=sigma,
            absolute_sigma=False,
            maxfev=10000,
        )
        sill = max(float(params[0]), 1e-6)
        range_ = float(-3.0 / params[1]) if params[1] != 0 else 40.0
        if range_ > 0.0 and np.isfinite(range_):
            return sill, 0.0, range_
    except (RuntimeError, ValueError, FloatingPointError):
        pass

    return 1.0, 0.0, 40.0


@njit(cache=True)
def _build_offsets(size_wind: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = (2 * size_wind + 1) * (2 * size_wind + 1) - 1
    dys = np.empty(n, dtype=np.int16)
    dxs = np.empty(n, dtype=np.int16)
    d2s = np.empty(n, dtype=np.int32)
    k = 0
    for dy in range(-size_wind, size_wind + 1):
        for dx in range(-size_wind, size_wind + 1):
            if dy == 0 and dx == 0:
                continue
            dys[k] = dy
            dxs[k] = dx
            d2s[k] = dy * dy + dx * dx
            k += 1
    # insertion sort by distance (n small: <= 624 for W=12)
    for i in range(1, n):
        yd = dys[i]; xd = dxs[i]; dd = d2s[i]
        j = i - 1
        while j >= 0 and d2s[j] > dd:
            dys[j + 1] = dys[j]
            dxs[j + 1] = dxs[j]
            d2s[j + 1] = d2s[j]
            j -= 1
        dys[j + 1] = yd
        dxs[j + 1] = xd
        d2s[j + 1] = dd
    return dys, dxs, d2s


@njit(cache=True)
def _cov_scalar(h: float, sill: float, nugget: float, range_: float) -> float:
    if h == 0.0:
        return sill + nugget
    return sill * np.exp(-3.0 * h / max(range_, 1e-6))


@njit(cache=True)
def _solve_linear_inplace(a: np.ndarray, b: np.ndarray, n: int) -> bool:
    """Small dense Gaussian elimination in-place for n x n system."""
    for k in range(n):
        piv = k
        maxv = abs(a[k, k])
        for r in range(k + 1, n):
            v = abs(a[r, k])
            if v > maxv:
                maxv = v
                piv = r
        if maxv < 1e-12:
            return False
        if piv != k:
            for c in range(k, n):
                tmp = a[k, c]
                a[k, c] = a[piv, c]
                a[piv, c] = tmp
            tmpb = b[k]
            b[k] = b[piv]
            b[piv] = tmpb
        inv = 1.0 / a[k, k]
        for c in range(k, n):
            a[k, c] *= inv
        b[k] *= inv
        for r in range(n):
            if r == k:
                continue
            factor = a[r, k]
            if factor == 0.0:
                continue
            for c in range(k, n):
                a[r, c] -= factor * a[k, c]
            b[r] -= factor * b[k]
    return True


@njit(parallel=True, cache=True)
def apply_gnspi_chunk_numba(
    rows: np.ndarray,
    cols: np.ndarray,
    sample_rows: np.ndarray,
    sample_cols: np.ndarray,
    sample_count: np.ndarray,
    mode: np.ndarray,
    cls_arr: np.ndarray,
    target: np.ndarray,
    inp: np.ndarray,
    series: np.ndarray,
    dif: np.ndarray,
    regress: np.ndarray,
    semiv: np.ndarray,
    out: np.ndarray,
    uncertainty: np.ndarray,
    quality: np.ndarray,
    trend: np.ndarray,
    residual_component: np.ndarray,
    sample_count_img: np.ndarray,
    min_krige_samples: int,
    residual_scale: float,
    dn_min: float,
    dn_max: float,
) -> None:
    n_pix = len(rows)
    bands = target.shape[0]
    sample_size = sample_rows.shape[1]
    n_series = series.shape[0]
    for i in prange(n_pix):
        yy = rows[i]
        xx = cols[i]
        cls = cls_arr[i]
        m = mode[i]
        cnt = sample_count[i]
        sample_count_img[yy, xx] = cnt
        if (m == 1 or m == 2) and cnt >= min_krige_samples:
            # Fixed max-size arrays; use first cnt entries and cnt+1 system.
            for b in range(bands):
                A = np.zeros((sample_size + 1, sample_size + 1), dtype=np.float64)
                rhs = np.zeros(sample_size + 1, dtype=np.float64)
                sill = float(semiv[0, b, cls])
                nugget = float(semiv[1, b, cls])
                range_ = float(semiv[2, b, cls])
                for a in range(cnt):
                    ya = sample_rows[i, a]
                    xa = sample_cols[i, a]
                    for bb in range(cnt):
                        yb = sample_rows[i, bb]
                        xb = sample_cols[i, bb]
                        h = ((ya - yb) * (ya - yb) + (xa - xb) * (xa - xb)) ** 0.5
                        A[a, bb] = _cov_scalar(h, sill, nugget, range_)
                    A[a, cnt] = 1.0
                    A[cnt, a] = 1.0
                    h0 = ((ya - yy) * (ya - yy) + (xa - xx) * (xa - xx)) ** 0.5
                    rhs[a] = _cov_scalar(h0, sill, nugget, range_)
                rhs[cnt] = 1.0
                rhs_orig = rhs.copy()
                ok = _solve_linear_inplace(A, rhs, cnt + 1)
                if ok:
                    kresid = 0.0
                    rhs_dot = 0.0
                    for a in range(cnt):
                        kresid += rhs[a] * dif[b, sample_rows[i, a], sample_cols[i, a]]
                        rhs_dot += rhs[a] * rhs_orig[a]
                    rhs_dot += rhs[cnt] * rhs_orig[cnt]
                    kvar = sill - rhs_dot
                else:
                    kresid = 0.0
                    for a in range(cnt):
                        kresid += dif[b, sample_rows[i, a], sample_cols[i, a]]
                    kresid /= max(cnt, 1)
                    kvar = 0.0
                trend_val = inp[b, yy, xx] * regress[b, cls, 1] + regress[b, cls, 0]
                kresid *= residual_scale
                pred = trend_val + kresid
                if pred < dn_min or pred > dn_max:
                    sm = 0.0
                    sc = 0
                    for s in range(n_series):
                        v = series[s, b, yy, xx]
                        if np.isfinite(v):
                            sm += v
                            sc += 1
                    pred = sm / sc if sc > 0 else np.nan
                out[b, yy, xx] = pred
                trend[b, yy, xx] = trend_val
                residual_component[b, yy, xx] = kresid
                uncertainty[b, yy, xx] = 196.0 * abs(residual_scale) * (max(kvar, 0.0) ** 0.5) / max(abs(pred), 1e-6)
            quality[yy, xx] = m
        elif m == 1 or m == 2:
            for b in range(bands):
                trend_val = inp[b, yy, xx] * regress[b, cls, 1] + regress[b, cls, 0]
                pred = trend_val
                if pred < dn_min or pred > dn_max:
                    sm = 0.0
                    sc = 0
                    for s in range(n_series):
                        v = series[s, b, yy, xx]
                        if np.isfinite(v):
                            sm += v
                            sc += 1
                    pred = sm / sc if sc > 0 else np.nan
                out[b, yy, xx] = pred
                trend[b, yy, xx] = trend_val
                residual_component[b, yy, xx] = pred - trend_val
                uncertainty[b, yy, xx] = 0.0
            quality[yy, xx] = m
        elif m == 3:
            for b in range(bands):
                predict1 = inp[b, yy, xx] * regress[b, cls, 1] + regress[b, cls, 0]
                predict2 = 0.0
                for a in range(cnt):
                    predict2 += target[b, sample_rows[i, a], sample_cols[i, a]]
                predict2 /= max(cnt, 1)
                pred = 0.5 * (predict1 + predict2)
                if pred < dn_min or pred > dn_max:
                    sm = 0.0
                    sc = 0
                    for s in range(n_series):
                        v = series[s, b, yy, xx]
                        if np.isfinite(v):
                            sm += v
                            sc += 1
                    pred = sm / sc if sc > 0 else np.nan
                out[b, yy, xx] = pred
                trend[b, yy, xx] = predict1
                residual_component[b, yy, xx] = pred - predict1
                uncertainty[b, yy, xx] = 0.0
            quality[yy, xx] = 3
        else:
            for b in range(bands):
                pred = inp[b, yy, xx] * regress[b, cls, 1] + regress[b, cls, 0]
                if pred < dn_min or pred > dn_max:
                    sm = 0.0
                    sc = 0
                    for s in range(n_series):
                        v = series[s, b, yy, xx]
                        if np.isfinite(v):
                            sm += v
                            sc += 1
                    pred = sm / sc if sc > 0 else np.nan
                out[b, yy, xx] = pred
                trend[b, yy, xx] = inp[b, yy, xx] * regress[b, cls, 1] + regress[b, cls, 0]
                residual_component[b, yy, xx] = pred - trend[b, yy, xx]
                uncertainty[b, yy, xx] = 0.0
            quality[yy, xx] = 4


@njit(parallel=True, cache=True)
def select_samples_chunk(
    gap_rows: np.ndarray,
    gap_cols: np.ndarray,
    labels: np.ndarray,
    gap: np.ndarray,
    target_valid: np.ndarray,
    input_valid: np.ndarray,
    series: np.ndarray,       # n_series,bands,h,w
    series_valid: np.ndarray, # n_series,h,w
    similar_th: np.ndarray,
    rmsd_divisor: float,
    dys: np.ndarray,
    dxs: np.ndarray,
    d2s: np.ndarray,
    sample_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_pix = len(gap_rows)
    n_offsets = len(dys)
    n_series = series.shape[0]
    bands = series.shape[1]
    h = labels.shape[0]
    w = labels.shape[1]
    sample_rows = np.full((n_pix, sample_size), -1, dtype=np.int32)
    sample_cols = np.full((n_pix, sample_size), -1, dtype=np.int32)
    sample_count = np.zeros(n_pix, dtype=np.int16)
    mode = np.zeros(n_pix, dtype=np.uint8)
    cls_out = np.zeros(n_pix, dtype=np.int16)

    for idx in prange(n_pix):
        y = gap_rows[idx]
        x = gap_cols[idx]
        cls = labels[y, x]
        cls_out[idx] = cls
        cnt = 0
        # Exact GNSPI selection: same class, not target gap, valid common pixels,
        # and intersection of similar pixels across all series images.
        for oi in range(n_offsets):
            yy = y + dys[oi]
            xx = x + dxs[oi]
            if yy < 0 or yy >= h or xx < 0 or xx >= w:
                continue
            if gap[yy, xx]:
                continue
            if labels[yy, xx] != cls:
                continue
            if not target_valid[yy, xx] or not input_valid[yy, xx]:
                continue
            ok = True
            for s in range(n_series):
                if not series_valid[s, yy, xx] or not series_valid[s, y, x]:
                    ok = False
                    break
                diff_sq = 0.0
                for b in range(bands):
                    d = series[s, b, yy, xx] - series[s, b, y, x]
                    diff_sq += d * d
                rmsd = (diff_sq / rmsd_divisor) ** 0.5
                if rmsd >= similar_th[s]:
                    ok = False
                    break
            if ok:
                sample_rows[idx, cnt] = yy
                sample_cols[idx, cnt] = xx
                cnt += 1
                if cnt >= sample_size:
                    break
        sample_count[idx] = cnt
        if cnt >= sample_size:
            mode[idx] = 1  # full GNSPI
        elif cnt > 0:
            mode[idx] = 2  # GNSPI fewer samples
        else:
            # fallback: mean of all nearest same-class valid target/input pixels
            found_d2 = -1
            tie_count = 0
            for oi in range(n_offsets):
                if found_d2 >= 0 and d2s[oi] > found_d2:
                    break
                yy = y + dys[oi]
                xx = x + dxs[oi]
                if yy < 0 or yy >= h or xx < 0 or xx >= w:
                    continue
                if gap[yy, xx]:
                    continue
                if labels[yy, xx] != cls:
                    continue
                if target_valid[yy, xx] and input_valid[yy, xx]:
                    if found_d2 < 0:
                        found_d2 = d2s[oi]
                    if d2s[oi] == found_d2 and tie_count < sample_size:
                        sample_rows[idx, tie_count] = yy
                        sample_cols[idx, tie_count] = xx
                        tie_count += 1
            if tie_count > 0:
                sample_count[idx] = tie_count
                mode[idx] = 3  # nearest same-class fallback
            else:
                mode[idx] = 4  # regression only
    return sample_rows, sample_cols, sample_count, mode, cls_out


def covariance_vec(h: np.ndarray, sill: float, nugget: float, range_: float) -> np.ndarray:
    cov = sill * np.exp(-3.0 * h / max(range_, 1e-6))
    cov = cov.astype(np.float64, copy=False)
    cov[h == 0.0] = sill + nugget
    return cov


def ordinary_kriging_bands(
    local_xy: np.ndarray,
    vals_by_band: np.ndarray,
    semiv_for_class: np.ndarray,
    x0: float,
    y0: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve ordinary kriging for all bands, reusing geometry distances."""
    m = local_xy.shape[0]
    bands = vals_by_band.shape[0]
    pred = np.zeros(bands, dtype=np.float32)
    var = np.zeros(bands, dtype=np.float32)
    if m == 0:
        return pred, var
    if m == 1:
        pred[:] = vals_by_band[:, 0]
        return pred, var

    pair_h = np.zeros((m, m), dtype=np.float64)
    target_h = np.zeros(m, dtype=np.float64)
    for i in range(m):
        target_h[i] = np.hypot(local_xy[i, 0] - y0, local_xy[i, 1] - x0)
        for j in range(m):
            pair_h[i, j] = np.hypot(local_xy[i, 0] - local_xy[j, 0], local_xy[i, 1] - local_xy[j, 1])

    for b in range(bands):
        sill = float(semiv_for_class[0, b])
        nugget = float(semiv_for_class[1, b])
        range_ = float(semiv_for_class[2, b])
        A = np.zeros((m + 1, m + 1), dtype=np.float64)
        A[:m, :m] = covariance_vec(pair_h, sill, nugget, range_)
        A[:m, m] = 1.0
        A[m, :m] = 1.0
        rhs = np.zeros(m + 1, dtype=np.float64)
        rhs[:m] = covariance_vec(target_h, sill, nugget, range_)
        rhs[m] = 1.0
        try:
            w = np.linalg.solve(A, rhs)
            weights = w[:m]
            pred[b] = float(np.sum(weights * vals_by_band[b]))
            var[b] = max(0.0, float(sill - np.dot(rhs, w)))
        except np.linalg.LinAlgError:
            pred[b] = float(np.mean(vals_by_band[b]))
            var[b] = 0.0
    return pred, var


def classwise_regression(target: np.ndarray, inp: np.ndarray, labels: np.ndarray, non_gap: np.ndarray, class_num: int) -> tuple[np.ndarray, np.ndarray]:
    bands = target.shape[0]
    regress = np.zeros((bands, class_num, 2), dtype=np.float32)
    dif = np.zeros_like(target, dtype=np.float32)
    for c in range(class_num):
        cmask = non_gap & (labels == c)
        if cmask.sum() < 2:
            regress[:, c, 0] = 0.0
            regress[:, c, 1] = 1.0
            continue
        for b in range(bands):
            x = inp[b, cmask].astype(np.float64)
            y = target[b, cmask].astype(np.float64)
            xm = x.mean(); ym = y.mean()
            denom = np.sum((x - xm) ** 2)
            slope = 1.0 if denom <= 1e-12 else np.sum((x - xm) * (y - ym)) / denom
            const = ym - slope * xm
            regress[b, c, 0] = const
            regress[b, c, 1] = slope
            dif[b, cmask] = target[b, cmask] - (inp[b, cmask] * slope + const)
    return regress, dif


def write_float_stack(path: Path, arr: np.ndarray, profile: dict, suffix: str) -> None:
    profile_out = profile.copy()
    profile_out.update(count=arr.shape[0], dtype="float32", nodata=np.nan, compress="deflate")
    with rasterio.open(path, "w", **profile_out) as dst:
        dst.write(arr.astype(np.float32))
        for idx, name in enumerate(BAND_NAMES, start=1):
            dst.set_band_description(idx, f"{name}_{suffix}")


def write_uint8_image(path: Path, arr: np.ndarray, profile: dict, description: str) -> None:
    profile_out = profile.copy()
    profile_out.update(count=1, dtype="uint8", nodata=255, compress="deflate")
    with rasterio.open(path, "w", **profile_out) as dst:
        dst.write(arr[None, :, :].astype(np.uint8))
        dst.set_band_description(1, description)


def fit_variograms(dif: np.ndarray, labels: np.ndarray, non_gap: np.ndarray, class_num: int) -> np.ndarray:
    bands, h, w = dif.shape
    ys, xs = np.indices((h, w))
    semiv = np.zeros((3, bands, class_num), dtype=np.float32)
    for c in range(class_num):
        cmask = non_gap & (labels == c)
        coords = np.column_stack([ys[cmask], xs[cmask]]).astype(np.float32)
        for b in range(bands):
            vals = dif[b, cmask].astype(np.float32)
            semiv[:, b, c] = fit_exponential_variogram(vals, coords)
    return semiv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--series", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--uncertainty-output", type=Path, default=None)
    parser.add_argument("--quality-output", type=Path, default=None)
    parser.add_argument("--class-num", type=int, default=4)
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--size-wind", type=int, default=12)
    parser.add_argument("--chunk-size", type=int, default=20000)
    parser.add_argument("--min-krige-samples", type=int, default=1)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--rmsd-divisor", type=float, default=0.0, help="0 means number of bands; IDL code used 3.0")
    parser.add_argument("--similar-threshold-all-pixels", action="store_true", help="Compute thresholds over all finite pixels, not valid-only")
    parser.add_argument("--idl-zero-mask", action="store_true", help="Use IDL zero-band rule for gap/non-gap")
    parser.add_argument("--kmeans-all-input-pixels", action="store_true", help="Classify all finite input pixels")
    parser.add_argument("--dn-min", type=float, default=-1.0)
    parser.add_argument("--dn-max", type=float, default=2.0)
    args = parser.parse_args()

    t0 = time.time()
    unc_path = args.uncertainty_output or args.output.with_name(args.output.stem + "_uncertainty.tif")
    qual_path = args.quality_output or args.output.with_name(args.output.stem + "_quality.tif")
    trend_path = args.output.with_name(args.output.stem + "_trend.tif")
    residual_path = args.output.with_name(args.output.stem + "_residual_component.tif")
    sample_count_path = args.output.with_name(args.output.stem + "_sample_count.tif")

    target, target_valid, profile = load_stack(args.target)
    inp, inp_valid, _ = load_stack(args.input)
    if args.idl_zero_mask:
        target_valid = ~(np.any(target == 0.0, axis=0) | ~np.isfinite(target).all(axis=0))
        inp_valid = ~(np.any(inp == 0.0, axis=0) | ~np.isfinite(inp).all(axis=0))
    series_list = [inp]
    series_valid_list = [inp_valid]
    for p in args.series:
        s, sv, _ = load_stack(p)
        series_list.append(s)
        series_valid_list.append(sv)
    series = np.stack(series_list).astype(np.float32)
    series_valid = np.stack(series_valid_list).astype(np.bool_)
    if args.idl_zero_mask:
        series_valid[:] = True
    bands, h, w = target.shape

    if args.idl_zero_mask:
        gap = (~target_valid) & inp_valid
        non_gap = ~gap
    else:
        gap = (~target_valid) & inp_valid
        non_gap = target_valid & inp_valid
    finite_input = np.isfinite(inp).all(axis=0)
    gap_rows, gap_cols = np.where(gap)
    print(f"gap pixels considered: {len(gap_rows)}", flush=True)

    print("kmeans start", flush=True)
    kmeans_mask = finite_input if args.kmeans_all_input_pixels else (inp_valid & finite_input)
    X = inp[:, kmeans_mask].T
    km = KMeans(n_clusters=args.class_num, random_state=0, n_init=10)
    labels = np.zeros((h, w), dtype=np.int16)
    labels[kmeans_mask] = km.fit_predict(X)
    print(f"kmeans done in {time.time()-t0:.1f}s", flush=True)

    regress, dif = classwise_regression(target, inp, labels, non_gap, args.class_num)
    print(f"regression done in {time.time()-t0:.1f}s", flush=True)

    semiv = fit_variograms(dif, labels, non_gap, args.class_num)
    print(f"semivariograms done in {time.time()-t0:.1f}s", flush=True)

    similar_th = []
    for s, sv in zip(series, series_valid):
        if args.similar_threshold_all_pixels:
            vals = s.reshape(s.shape[0], -1)
        else:
            vals = s[:, sv]
        th_band = np.nanstd(vals, axis=1) * 2.0 / float(args.class_num)
        similar_th.append(float(np.nanmean(th_band)))
    similar_th_arr = np.asarray(similar_th, dtype=np.float32)
    print("similar thresholds:", similar_th, flush=True)
    rmsd_divisor = float(bands if args.rmsd_divisor <= 0 else args.rmsd_divisor)
    print(f"rmsd divisor: {rmsd_divisor}", flush=True)

    dys, dxs, d2s = _build_offsets(args.size_wind)
    out = target.copy()
    uncertainty = np.zeros_like(target, dtype=np.float32)
    quality = np.zeros((h, w), dtype=np.uint16)
    trend = np.full_like(target, np.nan, dtype=np.float32)
    residual_component = np.full_like(target, np.nan, dtype=np.float32)
    sample_count_img = np.full((h, w), 255, dtype=np.uint8)

    sample_target_valid = np.ones_like(target_valid, dtype=np.bool_) if args.idl_zero_mask else target_valid
    sample_input_valid = np.ones_like(inp_valid, dtype=np.bool_) if args.idl_zero_mask else inp_valid

    total = len(gap_rows)
    for start in range(0, total, args.chunk_size):
        end = min(total, start + args.chunk_size)
        c0 = time.time()
        rows = gap_rows[start:end].astype(np.int32)
        cols = gap_cols[start:end].astype(np.int32)
        sample_rows, sample_cols, sample_count, mode, cls_arr = select_samples_chunk(
            rows, cols, labels, gap, sample_target_valid, sample_input_valid, series, series_valid,
            similar_th_arr, rmsd_divisor, dys, dxs, d2s, args.sample_size,
        )
        uniq, cnts = np.unique(mode, return_counts=True)
        mode_counts = {int(k): int(v) for k, v in zip(uniq, cnts)}
        print(
            f"chunk {start}:{end}/{total} samples selected in {time.time()-c0:.1f}s "
            f"mode_counts={mode_counts} sample_mean={float(sample_count.mean()):.2f} sample_max={int(sample_count.max())}",
            flush=True,
        )
        apply_gnspi_chunk_numba(
            rows,
            cols,
            sample_rows,
            sample_cols,
            sample_count,
            mode,
            cls_arr,
            target,
            inp,
            series,
            dif,
            regress,
            semiv,
            out,
            uncertainty,
            quality,
            trend,
            residual_component,
            sample_count_img,
            args.min_krige_samples,
            args.residual_scale,
            args.dn_min,
            args.dn_max,
        )
        print(f"chunk {start}:{end}/{total} applied in {time.time()-c0:.1f}s total elapsed {time.time()-t0:.1f}s", flush=True)

    write_outputs(args.output, qual_path, out, quality.astype(np.uint8), profile)
    write_float_stack(unc_path, uncertainty, profile, "uncertainty_pct")
    write_float_stack(trend_path, trend, profile, "trend")
    write_float_stack(residual_path, residual_component, profile, "residual_component")
    write_uint8_image(sample_count_path, sample_count_img, profile, "gnspi_sample_count")
    meta = {
        "target": str(args.target), "input": str(args.input), "series": [str(p) for p in args.series],
        "output": str(args.output), "uncertainty_output": str(unc_path), "quality_output": str(qual_path),
        "trend_output": str(trend_path), "residual_component_output": str(residual_path),
        "sample_count_output": str(sample_count_path),
        "class_num": args.class_num, "sample_size": args.sample_size, "size_wind": args.size_wind,
        "chunk_size": args.chunk_size, "min_krige_samples": args.min_krige_samples,
        "residual_scale": args.residual_scale, "rmsd_divisor": rmsd_divisor,
        "similar_threshold_all_pixels": args.similar_threshold_all_pixels,
        "idl_zero_mask": args.idl_zero_mask,
        "kmeans_all_input_pixels": args.kmeans_all_input_pixels,
        "similar_thresholds": similar_th,
        "semivariogram": semiv.tolist(), "runtime_sec": time.time() - t0,
    }
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {args.output}", flush=True)
    print(f"wrote {qual_path}", flush=True)
    print(f"wrote {unc_path}", flush=True)
    print(f"wrote {trend_path}", flush=True)
    print(f"wrote {residual_path}", flush=True)
    print(f"wrote {sample_count_path}", flush=True)


if __name__ == "__main__":
    main()
