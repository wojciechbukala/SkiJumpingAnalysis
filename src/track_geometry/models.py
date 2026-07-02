# Track geometry data models
# Step one validates input arrays for image curves correspondences and matrices
# Step two stores camera calibration and computes the inverse intrinsic matrix
# Step three converts image pixels and vanishing points into unit camera rays
# Step four stores image curves correspondence pairs observations quality values and reconstruction results
# Option camera can be absent for image only observations
# Option track_width controls metric reconstruction scale when available
# Option correspondences can provide manual matching instead of sampled matching
# Formula pixel ray is normalize(inv_K @ [u v 1])
# Formula vanishing direction is normalize(inv_K @ v)
# Formula has metric reconstruction when both left_3d and right_3d exist

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


# Convert values to image points
def _as_points2d(points, name: str) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must be an array-like object with shape (N, 2).")
    if len(arr) < 2:
        raise ValueError(f"{name} must contain at least two points.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


# Convert values to a matrix
def _as_matrix3x3(matrix, name: str) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3).")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


# Normalize a vector
def _unit(vector: np.ndarray, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError(f"{name} has near-zero norm.")
    return vector / norm


@dataclass
class CameraCalibration:
    # Store pinhole camera intrinsics

    K: np.ndarray

    # Validate camera calibration data
    def __post_init__(self) -> None:
        self.K = _as_matrix3x3(self.K, "K")
        self.inv_K = np.linalg.inv(self.K)

    # Create calibration from values
    @classmethod
    def from_sequence(cls, values) -> "CameraCalibration":
        return cls(_as_matrix3x3(values, "camera_matrix"))

    # Convert a pixel to a ray
    def pixel_to_ray(self, point: np.ndarray) -> np.ndarray:
        u, v = np.asarray(point, dtype=float)
        ray = self.inv_K @ np.array([u, v, 1.0], dtype=float)
        return _unit(ray, "camera ray")

    # Recover direction from a vanishing point
    def direction_from_vanishing_point(self, vanishing_point: np.ndarray) -> np.ndarray:
        v = np.asarray(vanishing_point, dtype=float).reshape(3)
        direction = self.inv_K @ v
        return _unit(direction, "vanishing direction")


@dataclass
class ImageCurve:
    # Store one visible border curve

    points: np.ndarray
    name: str = "curve"

    # Validate image curve data
    def __post_init__(self) -> None:
        self.points = _as_points2d(self.points, self.name)


@dataclass
class ImageCorrespondences:
    # Store paired image points

    left: np.ndarray
    right: np.ndarray

    # Validate correspondence data
    def __post_init__(self) -> None:
        self.left = _as_points2d(self.left, "left correspondences")
        self.right = _as_points2d(self.right, "right correspondences")
        if len(self.left) != len(self.right):
            raise ValueError("left and right correspondences must have the same length.")


@dataclass
class ParallelCurveObservation:
    # Store input geometric observations

    left_curve: ImageCurve
    right_curve: ImageCurve
    camera: CameraCalibration | None = None
    track_width: float | None = None
    correspondences: ImageCorrespondences | None = None
    image_path: Path | None = None
    name: str = "parallel_curve_observation"
    num_samples: int = 80

    # Validate observation settings
    def __post_init__(self) -> None:
        if self.track_width is not None and self.track_width <= 0:
            raise ValueError("track_width must be positive when provided.")
        if self.num_samples < 2:
            raise ValueError("num_samples must be at least 2.")


@dataclass
class ReconstructionQuality:
    mean_ruling_residual: float | None = None
    max_ruling_residual: float | None = None
    mean_vanishing_residual: float | None = None
    max_plane_error: float | None = None
    positive_depth_ratio: float | None = None


@dataclass
class ParallelCurveReconstruction:
    # Store reconstructed parallel curves

    image_left: np.ndarray
    image_right: np.ndarray
    vanishing_point: np.ndarray
    left_3d: np.ndarray | None = None
    right_3d: np.ndarray | None = None
    center_3d: np.ndarray | None = None
    plane_normal: np.ndarray | None = None
    track_width: float | None = None
    scale_is_metric: bool = False
    ruling_residuals: np.ndarray | None = None
    vanishing_residuals: np.ndarray | None = None
    quality: ReconstructionQuality | None = None
    message: str = ""

    # Report whether metric data exists
    @property
    def has_metric_reconstruction(self) -> bool:
        return self.left_3d is not None and self.right_3d is not None

    # Summarize reconstruction results
    def summary(self) -> str:
        if not self.has_metric_reconstruction:
            return self.message or "Only image correspondences and vanishing geometry are available."

        width_label = "metric" if self.scale_is_metric else "relative"
        quality = self.quality or ReconstructionQuality()
        mean_residual = quality.mean_ruling_residual
        residual_text = "n/a" if mean_residual is None else f"{mean_residual:.6g}"
        return (
            f"Reconstructed {len(self.left_3d)} corresponding 3D points "
            f"with {width_label} scale; mean ruling residual: {residual_text}."
        )
