from __future__ import annotations

import numpy as np


def smooth_trajectory(
    traj: list[tuple[float, float, float]],
    window_size: int = 15,
    poly_order: int = 2,
) -> list[tuple[float, float, float]]:
    if not traj or len(traj) < window_size or poly_order >= window_size:
        return traj

    xs = np.asarray([point[0] for point in traj], dtype=float)
    ys = np.asarray([point[1] for point in traj], dtype=float)
    frames = [point[2] for point in traj]

    try:
        from scipy.signal import savgol_filter

        smoothed_x = savgol_filter(xs, window_size, poly_order, mode="nearest")
        smoothed_y = savgol_filter(ys, window_size, poly_order, mode="nearest")
    except ImportError:
        smoothed_x = _moving_average(xs, window_size)
        smoothed_y = _moving_average(ys, window_size)

    return list(zip(smoothed_x, smoothed_y, frames))


def _moving_average(values: np.ndarray, window_size: int) -> np.ndarray:
    radius = window_size // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    kernel = np.ones(window_size, dtype=float) / float(window_size)
    return np.convolve(padded, kernel, mode="valid")
