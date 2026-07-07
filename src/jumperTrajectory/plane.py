from __future__ import annotations

import numpy as np

from src.jumperTrajectory.stabilization import apply_homography_to_points
from viz_slope import build_plane_homography, unit


def build_center_plane_homography(bundle: dict, K: np.ndarray) -> dict:
    origin_center = np.asarray(bundle.get("origin_center", 0.5 * (bundle["origin_a"] + bundle["origin_b"])), dtype=float)
    e1 = unit(bundle["e1"], "asse u piano centrale")
    e2 = unit(bundle["e2"], "asse v piano centrale")
    normal = unit(bundle["normal"], "normale piano centrale")
    if "H_center_plane_to_image" in bundle and "H_image_to_center_plane" in bundle:
        H_plane_to_image = np.asarray(bundle["H_center_plane_to_image"], dtype=float)
        H_image_to_plane = np.asarray(bundle["H_image_to_center_plane"], dtype=float)
    else:
        H_plane_to_image, H_image_to_plane = build_plane_homography(K, origin_center, e1, e2)
    return {
        "origin_center": origin_center,
        "e1": e1,
        "e2": e2,
        "normal": normal,
        "H_center_plane_to_image": H_plane_to_image,
        "H_image_to_center_plane": H_image_to_plane,
    }


def lift_points_to_center_plane(reference_points: np.ndarray, center_plane: dict) -> tuple[np.ndarray, np.ndarray]:
    coords = apply_homography_to_points(center_plane["H_image_to_center_plane"], reference_points)
    points_3d = (
        center_plane["origin_center"]
        + coords[:, :1] * center_plane["e1"]
        + coords[:, 1:2] * center_plane["e2"]
    )
    return coords, points_3d
