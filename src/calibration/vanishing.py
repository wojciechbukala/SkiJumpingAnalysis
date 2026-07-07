from __future__ import annotations

import numpy as np

from src.track_geometry.calibration import CalibrationError


def line_from_segment(segment: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = [float(value) for value in segment]
    line = np.array([y1 - y2, x2 - x1, x1 * y2 - x2 * y1], dtype=float)
    norm = float(np.linalg.norm(line[:2]))
    if norm < 1e-12:
        raise ValueError("Linea degenerata.")
    return line / norm


def vanishing_point_from_line_group(segments: list[tuple[int, int, int, int]]) -> list[float]:
    line_matrix = np.vstack([line_from_segment(segment) for segment in segments])
    _, _, vh = np.linalg.svd(line_matrix)
    point = vh[-1, :]

    if abs(float(point[2])) < 1e-9:
        raise CalibrationError("Il gruppo di linee produce un punto di fuga quasi all'infinito.")

    point = point / point[2]
    if not np.all(np.isfinite(point)):
        raise CalibrationError("Punto di fuga non finito.")

    return [float(point[0]), float(point[1])]


def vanishing_points_from_line_groups(
    lines: list[tuple[int, int, int, int]],
    selected_groups: list[list[int]],
) -> list[list[float]]:
    return [
        vanishing_point_from_line_group([lines[idx] for idx in group])
        for group in selected_groups
    ]


def print_vanishing_points(selected_groups: list[list[int]], vanishing_points: list[list[float]]) -> None:
    print("Punti di fuga stimati:")
    for group_idx, (group, point) in enumerate(zip(selected_groups, vanishing_points), start=1):
        print(f"  Gruppo {group_idx} linee {group}: x={point[0]:.6f}, y={point[1]:.6f}")
