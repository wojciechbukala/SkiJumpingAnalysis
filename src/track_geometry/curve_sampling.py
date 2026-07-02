# Curve sampling operations
# Step one removes consecutive duplicate vertices so length calculations are stable
# Step two computes cumulative arc length along each polyline
# Step three resamples each curve at evenly spaced target lengths
# Step four builds left and right image correspondences for parallel curve reconstruction
# Option num_samples overrides the observation sample count
# Option explicit correspondences preserves manually provided matches unless resampling is requested
# Formula segment length is sqrt(dx^2 + dy^2)
# Formula cumulative length is s_i = sum of previous segment lengths
# Formula resampling uses linear interpolation at target lengths from zero to total length

from __future__ import annotations

import numpy as np

from .models import ImageCorrespondences, ParallelCurveObservation


# Remove repeated curve vertices
def remove_duplicate_vertices(points: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    # Drop consecutive duplicate vertices
    points = np.asarray(points, dtype=float)
    if len(points) <= 1:
        return points

    keep = [0]
    for idx in range(1, len(points)):
        if np.linalg.norm(points[idx] - points[keep[-1]]) > eps:
            keep.append(idx)
    return points[keep]


# Compute cumulative curve length
def cumulative_arclength(points: np.ndarray) -> np.ndarray:
    points = remove_duplicate_vertices(points)
    if len(points) < 2:
        raise ValueError("A curve must contain at least two distinct points.")

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    if cumulative[-1] <= 1e-12:
        raise ValueError("A curve with zero length cannot be resampled.")
    return cumulative


# Sample a polyline uniformly
def resample_polyline(points: np.ndarray, num_samples: int) -> np.ndarray:
    # Sample a polyline uniformly by image length
    if num_samples < 2:
        raise ValueError("num_samples must be at least 2.")

    points = remove_duplicate_vertices(points)
    cumulative = cumulative_arclength(points)
    targets = np.linspace(0.0, cumulative[-1], num_samples)
    x = np.interp(targets, cumulative, points[:, 0])
    y = np.interp(targets, cumulative, points[:, 1])
    return np.column_stack([x, y])


# Match samples across parallel curves
def build_correspondences(
    observation: ParallelCurveObservation,
    num_samples: int | None = None,
) -> ImageCorrespondences:
    # Produce corresponding image points
    # Preserve provided matches when available
    # Sample ordered curves with a common parameter
    if observation.correspondences is not None:
        if num_samples is None:
            return observation.correspondences
        left = resample_polyline(observation.correspondences.left, num_samples)
        right = resample_polyline(observation.correspondences.right, num_samples)
        return ImageCorrespondences(left=left, right=right)

    sample_count = num_samples or observation.num_samples
    left = resample_polyline(observation.left_curve.points, sample_count)
    right = resample_polyline(observation.right_curve.points, sample_count)
    return ImageCorrespondences(left=left, right=right)
