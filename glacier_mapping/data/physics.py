import numpy as np
import numba

import glacier_mapping.utils.logging as log

neighbor_offsets = np.array(
    [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
    ],
    dtype=np.int32,
)


@numba.njit()
def bfs_fast(im, water_allpath, source, queue_r, queue_c, visited):
    rows, cols = im.shape

    for i in range(rows):
        for j in range(cols):
            visited[i, j] = False

    sr, sc = source

    queue_start = 0
    queue_end = 1
    queue_r[0] = sr
    queue_c[0] = sc
    visited[sr, sc] = True

    while queue_start < queue_end:
        ur = queue_r[queue_start]
        uc = queue_c[queue_start]
        queue_start += 1

        curr_elev = im[ur, uc]

        if not (ur == sr and uc == sc):
            water_allpath[ur, uc] += 1.0

        for k in range(8):
            vr = ur + neighbor_offsets[k, 0]
            vc = uc + neighbor_offsets[k, 1]

            if vr < 0 or vr >= rows or vc < 0 or vc >= cols:
                continue

            if not visited[vr, vc] and im[vr, vc] < curr_elev:
                visited[vr, vc] = True
                queue_r[queue_end] = vr
                queue_c[queue_end] = vc
                queue_end += 1

    return water_allpath


def resize(arr: np.ndarray, new_rows: int, new_cols: int) -> np.ndarray:
    import cv2

    arr = arr.astype(np.float32, copy=False)
    resized = cv2.resize(arr, (new_cols, new_rows), interpolation=cv2.INTER_NEAREST)
    return resized.astype(np.float32)


def compute_flow(elevation, res=64, scale=0.3):
    if elevation.ndim == 3:
        elev_2d = elevation[:, :, 0].astype(np.float32)
    else:
        elev_2d = elevation.astype(np.float32)

    original_shape = elev_2d.shape

    if scale != 1.0:
        new_rows = int(original_shape[0] * scale)
        new_cols = int(original_shape[1] * scale)
        elev_2d = resize(elev_2d, new_rows, new_cols)

    rows, cols = elev_2d.shape

    water_allpath = np.zeros((rows, cols), dtype=np.float32)

    queue_r = np.zeros(rows * cols, dtype=np.int32)
    queue_c = np.zeros(rows * cols, dtype=np.int32)
    visited = np.zeros((rows, cols), dtype=np.bool_)

    if res == "full":
        res = 1

    step = int(res)

    for i in range(0, rows, step):
        for j in range(0, cols, step):
            bfs_fast(elev_2d, water_allpath, (i, j), queue_r, queue_c, visited)

    if scale != 1.0:
        water_allpath = resize(water_allpath, original_shape[0], original_shape[1])

    return water_allpath.astype(np.float32)


@numba.njit()
def uniform_filter_numba(arr, radius):
    rows, cols = arr.shape
    result = np.zeros_like(arr)
    half = radius // 2

    for i in range(rows):
        for j in range(cols):
            total = 0.0
            count = 0
            for di in range(-half, half + 1):
                for dj in range(-half, half + 1):
                    ni = i + di
                    nj = j + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        total += arr[ni, nj]
                        count += 1
            result[i, j] = total / count
    return result


def compute_tpi(elevation: np.ndarray, radius: int = 5) -> np.ndarray:
    elevation = elevation.astype(np.float32, copy=False)

    try:
        mean_elev = uniform_filter_numba(elevation, radius)
    except Exception:
        from scipy.ndimage import uniform_filter

        mean_elev = uniform_filter(elevation, size=radius)

    tpi = elevation - mean_elev
    return tpi.astype(np.float32)


@numba.njit()
def rolling_std_numba(arr, window):
    rows, cols = arr.shape
    result = np.zeros_like(arr)
    half = window // 2

    padded = np.zeros((rows + 2 * half, cols + 2 * half), dtype=arr.dtype)
    padded[half : half + rows, half : half + cols] = arr

    for i in range(rows):
        for j in range(cols):
            wsum = 0.0
            count = 0
            for di in range(window):
                for dj in range(window):
                    v = padded[i + di, j + dj]
                    wsum += v
                    count += 1
            mean = wsum / count

            var = 0.0
            for di in range(window):
                for dj in range(window):
                    v = padded[i + di, j + dj]
                    diff = v - mean
                    var += diff * diff
            var /= count

            result[i, j] = np.sqrt(var)

    return result


def compute_roughness(elevation: np.ndarray, window: int = 3) -> np.ndarray:
    elevation = elevation.astype(np.float32, copy=False)

    try:
        rough = rolling_std_numba(elevation, window)
    except Exception:
        from scipy.ndimage import generic_filter

        rough = generic_filter(elevation, np.std, size=window)

    return rough.astype(np.float32)


def compute_plan_curvature(elevation: np.ndarray) -> np.ndarray:
    elevation = elevation.astype(np.float32, copy=False)

    dy, dx = np.gradient(elevation)
    dxy, dxx = np.gradient(dx)
    dyy, dyx = np.gradient(dy)

    numerator = dxx * dy**2 - 2.0 * dxy * dx * dy + dyy * dx**2
    denominator = (dx**2 + dy**2 + 1e-8) ** 1.5 + 1e-8
    curv = numerator / denominator

    return curv.astype(np.float32)


def compute_phys_v4(elevation_full: np.ndarray, res=64, scale=1.0) -> np.ndarray:
    if elevation_full.ndim == 3:
        elevation = elevation_full[:, :, 0].astype(np.float32)
    else:
        elevation = elevation_full.astype(np.float32)

    flow = compute_flow(elevation, res=res, scale=scale)
    tpi = compute_tpi(elevation, radius=5)
    rough = compute_roughness(elevation, window=3)
    curv = compute_plan_curvature(elevation)

    return np.stack(
        [
            flow.astype(np.float32),
            tpi.astype(np.float32),
            rough.astype(np.float32),
            curv.astype(np.float32),
        ],
        axis=-1,
    )


if __name__ == "__main__":
    import glacier_mapping.data.slice as fn

    dem = fn.read_tiff("/data/baryal/HKH/DEM/image1.tif")
    dem_np = np.transpose(dem.read(), (1, 2, 0)).astype(np.float32)
    dem_np = np.nan_to_num(dem_np)
    elevation = dem_np[:, :, 0][:, :, None]

    phys_output = compute_phys_v4(elevation, res=64, scale=0.3)
    log.debug(f"Physics output shape: {phys_output.shape}, dtype: {phys_output.dtype}")
