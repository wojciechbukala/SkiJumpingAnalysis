from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.jumperTrajectory.motion import normalize_homography


def apply_homography_to_points(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    points_h = np.column_stack([points, np.ones(len(points), dtype=float)])
    mapped = (H @ points_h.T).T
    valid = np.abs(mapped[:, 2]) > 1e-12
    output = np.full((len(points), 2), np.nan, dtype=float)
    output[valid] = mapped[valid, :2] / mapped[valid, 2:3]
    return output


def read_kalman_points(csv_path: Path, start_frame: int, end_frame: int) -> dict[int, tuple[float, float]]:
    points: dict[int, tuple[float, float]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"frame", "kf_x", "kf_y"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV incompleto, mancano colonne: {sorted(missing)}")

        for row in reader:
            frame = int(float(row["frame"]))
            if frame < start_frame or frame > end_frame:
                continue
            x = float(row["kf_x"])
            y = float(row["kf_y"])
            if np.isfinite(x) and np.isfinite(y):
                points[frame] = (x, y)

    if not points:
        raise ValueError(f"Nessun punto Kalman valido nel range {start_frame}-{end_frame}.")
    return points


def compute_frame_to_reference_transforms(
    pairwise_warps: dict[int, np.ndarray],
    start_frame: int,
    end_frame: int,
    reference_frame: int,
) -> dict[int, np.ndarray]:
    if not start_frame <= reference_frame <= end_frame:
        raise ValueError("Il frame di riferimento deve stare dentro l'intervallo camera.")

    transforms: dict[int, np.ndarray] = {reference_frame: np.eye(3, dtype=float)}

    for frame in range(reference_frame + 1, end_frame + 1):
        if frame not in pairwise_warps:
            raise ValueError(f"Manca omografia consecutiva {frame}->{frame - 1}.")
        transforms[frame] = normalize_homography(transforms[frame - 1] @ pairwise_warps[frame])

    for frame in range(reference_frame - 1, start_frame - 1, -1):
        next_frame = frame + 1
        if next_frame not in pairwise_warps:
            raise ValueError(f"Manca omografia consecutiva {next_frame}->{frame}.")
        transforms[frame] = normalize_homography(transforms[next_frame] @ np.linalg.inv(pairwise_warps[next_frame]))

    return transforms


def compute_cumulative_motion_scores(
    motion_rows: list[dict],
    start_frame: int,
    end_frame: int,
    reference_frame: int,
) -> dict[int, float]:
    pair_scores = {int(row["frame"]): float(row["score"]) for row in motion_rows}
    scores = {reference_frame: 1.0}

    for frame in range(reference_frame + 1, end_frame + 1):
        scores[frame] = min(scores[frame - 1], pair_scores.get(frame, 0.0))

    for frame in range(reference_frame - 1, start_frame - 1, -1):
        scores[frame] = min(scores[frame + 1], pair_scores.get(frame + 1, 0.0))

    return scores


def stabilize_kalman_points(
    kalman_points: dict[int, tuple[float, float]],
    transforms: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = np.array(sorted(frame for frame in kalman_points if frame in transforms), dtype=int)
    if len(frames) == 0:
        raise ValueError("Nessun frame ha sia punto Kalman sia omografia verso il riferimento.")

    image_points = np.array([kalman_points[int(frame)] for frame in frames], dtype=float)
    reference_points = np.vstack(
        [
            apply_homography_to_points(transforms[int(frame)], image_points[idx : idx + 1])[0]
            for idx, frame in enumerate(frames)
        ],
    )
    valid = np.all(np.isfinite(reference_points), axis=1)
    return frames[valid], image_points[valid], reference_points[valid]
