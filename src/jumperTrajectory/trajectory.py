from __future__ import annotations

import numpy as np


def find_jumper_point(bbox: np.ndarray) -> tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def _apply_homogeneous_transform(matrix: np.ndarray, point: tuple[float, float]) -> tuple[float, float, float] | None:
    mapped = np.asarray(matrix, dtype=float) @ np.array([point[0], point[1], 1.0], dtype=float)
    if abs(float(mapped[2])) < 1e-12:
        return None
    return float(mapped[0] / mapped[2]), float(mapped[1] / mapped[2]), 1.0


def compute_trajectory(transformations, ref_frame: int, stop_frame: int, provider) -> list[tuple[float, float, float]]:
    trajectory: list[tuple[float, float, float]] = []
    if stop_frame <= ref_frame:
        return trajectory

    use_edges = len(transformations) > 0 and _is_edge_like(transformations[0])
    warp_by_pair = {}
    if use_edges:
        warp_by_pair = {(edge.frame_i_index, edge.frame_j_index): edge.warp_matrix for edge in transformations}

    cumulative = np.eye(3, dtype=np.float64)
    for local_idx, frame_global in enumerate(range(ref_frame, stop_frame)):
        if use_edges:
            if local_idx > 0:
                warp = warp_by_pair.get((local_idx, local_idx - 1))
                if warp is not None:
                    cumulative = cumulative @ warp
            transform = cumulative
        else:
            transform = transformations[local_idx]

        point = provider.get_point(frame_global)
        if point is None:
            continue

        mapped = _apply_homogeneous_transform(transform, point)
        if mapped is not None:
            trajectory.append(mapped)

    return trajectory


def _is_edge_like(value) -> bool:
    return all(
        hasattr(value, name)
        for name in ("frame_i_index", "frame_j_index", "warp_matrix")
    )
