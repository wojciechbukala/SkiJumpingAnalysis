from __future__ import annotations

import numpy as np

from src.track_geometry.curve_sampling import resample_polyline
from src.track_geometry.models import (
    CameraCalibration,
    ImageCorrespondences,
    ParallelCurveObservation,
    ParallelCurveReconstruction,
    ReconstructionQuality,
)


def unit(vector: np.ndarray, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError(f"{name} ha norma quasi nulla.")
    return vector / norm


def line_from_points(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    x1, y1 = [float(value) for value in first]
    x2, y2 = [float(value) for value in second]
    line = np.array([y1 - y2, x2 - x1, x1 * y2 - x2 * y1], dtype=float)
    return unit(line, "linea immagine")


def line_from_segment(segment: tuple[tuple[float, float], tuple[float, float]]) -> np.ndarray:
    return line_from_points(np.asarray(segment[0], dtype=float), np.asarray(segment[1], dtype=float))


def estimate_vanishing_point_from_segments(
    segments: list[tuple[tuple[float, float], tuple[float, float]]],
    min_segments: int = 2,
) -> np.ndarray:
    if len(segments) < min_segments:
        raise ValueError(f"Servono almeno {min_segments} segmenti.")

    line_matrix = np.vstack([line_from_segment(segment) for segment in segments])
    _, _, vh = np.linalg.svd(line_matrix)
    point = vh[-1, :]
    if abs(float(point[2])) > 1e-12:
        point = point / point[2]
    else:
        point = point / np.linalg.norm(point[:2])
    return point.astype(float)


def camera_direction_from_vanishing_point(K: np.ndarray, vanishing_point: np.ndarray) -> np.ndarray:
    direction = np.linalg.inv(K) @ np.asarray(vanishing_point, dtype=float).reshape(3)
    return unit(direction, "direzione normale")


def homogeneous_line_distance(line: np.ndarray, points: np.ndarray) -> np.ndarray:
    denom = float(np.linalg.norm(line[:2]))
    if denom < 1e-12:
        raise ValueError("Linea omogenea degenerata.")
    return np.abs(points @ line[:2] + line[2]) / denom


def line_through_point_and_vanishing(point: np.ndarray, vanishing_point: np.ndarray) -> np.ndarray:
    point_h = np.array([point[0], point[1], 1.0], dtype=float)
    line = np.cross(point_h, vanishing_point)
    return unit(line, "linea di corrispondenza")


def _dense_resample(points: np.ndarray, num_samples: int) -> np.ndarray:
    count = max(num_samples * 20, len(points) * 40)
    return resample_polyline(np.asarray(points, dtype=float), count)


def _monotonic_match_curve(
    points_a: np.ndarray,
    dense_b: np.ndarray,
    vanishing_point: np.ndarray,
    regularization_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(points_a)
    m = len(dense_b)
    if m < n:
        dense_b = resample_polyline(dense_b, n)
        m = len(dense_b)

    costs = np.zeros((n, m), dtype=float)
    raw_distances = np.zeros((n, m), dtype=float)
    normalized_arc = np.linspace(0.0, 1.0, m)

    for idx, point in enumerate(points_a):
        line = line_through_point_and_vanishing(point, vanishing_point)
        distances = homogeneous_line_distance(line, dense_b)
        target = idx / max(1, n - 1)
        costs[idx] = distances + regularization_px * np.abs(normalized_arc - target)
        raw_distances[idx] = distances

    dp = np.full((n, m), np.inf, dtype=float)
    back = np.full((n, m), -1, dtype=int)
    dp[0] = costs[0]

    for i in range(1, n):
        best_value = np.inf
        best_index = -1
        for j in range(m):
            if j > 0 and dp[i - 1, j - 1] < best_value:
                best_value = dp[i - 1, j - 1]
                best_index = j - 1
            if best_index >= 0:
                dp[i, j] = costs[i, j] + best_value
                back[i, j] = best_index

    end = int(np.argmin(dp[-1]))
    if not np.isfinite(dp[-1, end]):
        indices = np.linspace(0, m - 1, n).round().astype(int)
        return dense_b[indices], raw_distances[np.arange(n), indices]

    indices = np.empty(n, dtype=int)
    indices[-1] = end
    for i in range(n - 1, 0, -1):
        indices[i - 1] = back[i, indices[i]]

    return dense_b[indices], raw_distances[np.arange(n), indices]


def build_image_correspondences_from_vanishing_point(
    curve_a: np.ndarray,
    curve_b: np.ndarray,
    vanishing_point: np.ndarray,
    num_samples: int,
    regularization_px: float = 8.0,
) -> tuple[ImageCorrespondences, np.ndarray, bool]:
    points_a = resample_polyline(np.asarray(curve_a, dtype=float), num_samples)
    candidates = []

    for reversed_curve in (False, True):
        source_b = np.asarray(curve_b, dtype=float)
        if reversed_curve:
            source_b = source_b[::-1]
        dense_b = _dense_resample(source_b, num_samples)
        points_b, errors = _monotonic_match_curve(points_a, dense_b, vanishing_point, regularization_px)
        candidates.append((float(np.mean(errors)), ImageCorrespondences(points_a, points_b), errors, reversed_curve))

    candidates.sort(key=lambda item: item[0])
    _, correspondences, errors, reversed_curve = candidates[0]
    return correspondences, errors, reversed_curve


def reconstruct_parallel_curves(observation: ParallelCurveObservation) -> ParallelCurveReconstruction:
    if observation.camera is None:
        raise ValueError("Metric reconstruction requires a camera calibration.")
    if observation.track_width is None:
        raise ValueError("Metric reconstruction requires a positive track width.")

    if observation.correspondences is None:
        vanishing_point = estimate_ruling_vanishing_point(
            ImageCorrespondences(observation.left_curve.points, observation.right_curve.points),
        )
        correspondences, vanishing_residuals, _ = build_image_correspondences_from_vanishing_point(
            observation.left_curve.points,
            observation.right_curve.points,
            vanishing_point,
            observation.num_samples,
        )
    else:
        correspondences = observation.correspondences
        vanishing_point = estimate_ruling_vanishing_point(correspondences)
        vanishing_residuals = ruling_line_distances(correspondences, vanishing_point)

    normal = observation.camera.direction_from_vanishing_point(vanishing_point)
    left_3d, right_3d, residuals, depths, oriented_normal = reconstruct_corresponding_rays(
        observation.camera,
        correspondences.left,
        correspondences.right,
        normal,
        observation.track_width,
    )
    residual_norms = np.linalg.norm(residuals, axis=1)
    plane_errors = plane_offset_errors(left_3d, right_3d, oriented_normal)

    return ParallelCurveReconstruction(
        image_left=correspondences.left,
        image_right=correspondences.right,
        vanishing_point=vanishing_point,
        left_3d=left_3d,
        right_3d=right_3d,
        center_3d=0.5 * (left_3d + right_3d),
        plane_normal=oriented_normal,
        track_width=observation.track_width,
        scale_is_metric=True,
        ruling_residuals=residuals,
        vanishing_residuals=vanishing_residuals,
        quality=ReconstructionQuality(
            mean_ruling_residual=float(np.mean(residual_norms)),
            max_ruling_residual=float(np.max(residual_norms)),
            mean_vanishing_residual=float(np.mean(vanishing_residuals)),
            max_plane_error=float(np.max(plane_errors)),
            positive_depth_ratio=float(np.mean(depths > 0.0)),
        ),
        message="OK",
    )


def estimate_ruling_vanishing_point(correspondences: ImageCorrespondences) -> np.ndarray:
    segments = [
        ((float(a[0]), float(a[1])), (float(b[0]), float(b[1])))
        for a, b in zip(correspondences.left, correspondences.right)
    ]
    return estimate_vanishing_point_from_segments(segments, min_segments=2)


def ruling_line_distances(correspondences: ImageCorrespondences, vanishing_point: np.ndarray) -> np.ndarray:
    errors = []
    for point_a, point_b in zip(correspondences.left, correspondences.right):
        line = line_from_points(point_a, point_b)
        if abs(float(vanishing_point[2])) > 1e-12:
            distance = abs(float(line @ vanishing_point)) / abs(float(vanishing_point[2]))
        else:
            distance = abs(float(line @ vanishing_point))
        errors.append(distance)
    return np.asarray(errors, dtype=float)


def reconstruct_corresponding_rays(
    camera: CameraCalibration,
    points_a: np.ndarray,
    points_b: np.ndarray,
    normal: np.ndarray,
    width: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    normal = unit(normal, "normale")
    rays_a = np.vstack([camera.pixel_to_ray(point) for point in points_a])
    rays_b = np.vstack([camera.pixel_to_ray(point) for point in points_b])
    candidates = []

    for sign in (1.0, -1.0):
        oriented_normal = sign * normal
        denom_a = rays_a @ oriented_normal
        denom_b = rays_b @ oriented_normal
        if np.any(np.abs(denom_a) < 1e-10) or np.any(np.abs(denom_b) < 1e-10):
            continue

        lhs = []
        rhs = []
        for ray_a, ray_b, dot_a, dot_b in zip(rays_a, rays_b, denom_a, denom_b):
            lhs.extend((ray_b / dot_b - ray_a / dot_a).tolist())
            rhs.extend((float(width) * (oriented_normal - ray_b / dot_b)).tolist())

        lhs_arr = np.asarray(lhs, dtype=float).reshape(-1, 1)
        rhs_arr = np.asarray(rhs, dtype=float)
        plane_offset, _, _, _ = np.linalg.lstsq(lhs_arr, rhs_arr, rcond=None)
        plane_offset = float(plane_offset[0])

        depth_a = plane_offset / denom_a
        depth_b = (plane_offset + float(width)) / denom_b
        left_3d = rays_a * depth_a[:, None]
        right_3d = rays_b * depth_b[:, None]
        residuals = (right_3d - left_3d) - float(width) * oriented_normal
        depths = np.concatenate([depth_a, depth_b])
        positive_depth_ratio = float(np.mean(depths > 0.0))
        mean_residual = float(np.mean(np.linalg.norm(residuals, axis=1)))
        candidates.append((positive_depth_ratio, -mean_residual, left_3d, right_3d, residuals, depths, oriented_normal))

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    if not candidates:
        raise ValueError("Ricostruzione impossibile: raggi quasi paralleli ai piani stimati.")
    _, _, left_3d, right_3d, residuals, depths, oriented_normal = candidates[0]
    return left_3d, right_3d, residuals, depths, oriented_normal


def plane_offset_errors(left_3d: np.ndarray, right_3d: np.ndarray, normal: np.ndarray) -> np.ndarray:
    left_offsets = left_3d @ normal
    right_offsets = right_3d @ normal
    return np.concatenate(
        [
            np.abs(left_offsets - np.mean(left_offsets)),
            np.abs(right_offsets - np.mean(right_offsets)),
        ],
    )
