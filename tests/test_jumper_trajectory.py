from __future__ import annotations

import unittest

import numpy as np

from jumper_trajectory import (
    apply_homography_to_points,
    build_center_plane_homography,
    compute_frame_to_reference_transforms,
    lift_points_to_center_plane,
)


def _translation(tx: float, ty: float) -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, tx],
            [0.0, 1.0, ty],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _project(K: np.ndarray, points_3d: np.ndarray) -> np.ndarray:
    projected = (K @ points_3d.T).T
    return projected[:, :2] / projected[:, 2:3]


class JumperTrajectoryTest(unittest.TestCase):
    def test_frame_to_reference_transforms_before_and_after_reference(self) -> None:
        pairwise = {
            11: _translation(-10.0, 2.0),
            12: _translation(-10.0, 2.0),
            13: _translation(-10.0, 2.0),
            14: _translation(-10.0, 2.0),
        }

        transforms = compute_frame_to_reference_transforms(pairwise, 10, 14, 12)

        point = np.array([[100.0, 50.0]], dtype=float)
        mapped_10 = apply_homography_to_points(transforms[10], point)[0]
        mapped_12 = apply_homography_to_points(transforms[12], point)[0]
        mapped_14 = apply_homography_to_points(transforms[14], point)[0]

        self.assertTrue(np.allclose(mapped_10, [120.0, 46.0]))
        self.assertTrue(np.allclose(mapped_12, [100.0, 50.0]))
        self.assertTrue(np.allclose(mapped_14, [80.0, 54.0]))

    def test_center_plane_homography_lifts_projected_points(self) -> None:
        K = np.array(
            [
                [1200.0, 0.0, 960.0],
                [0.0, 900.0, 540.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        origin_a = np.array([0.4, -0.2, 8.0], dtype=float)
        normal = np.array([0.35, 0.05, 0.935], dtype=float)
        normal = normal / np.linalg.norm(normal)
        width = 1.2
        origin_b = origin_a + width * normal
        e1 = np.array([normal[2], 0.0, -normal[0]], dtype=float)
        e1 = e1 / np.linalg.norm(e1)
        e2 = np.cross(normal, e1)
        e2 = e2 / np.linalg.norm(e2)
        bundle = {
            "origin_a": origin_a,
            "origin_b": origin_b,
            "e1": e1,
            "e2": e2,
            "normal": normal,
        }
        center_plane = build_center_plane_homography(bundle, K)
        expected_coords = np.array(
            [
                [0.0, 0.0],
                [1.5, -0.3],
                [-0.7, 0.9],
                [2.2, 1.1],
            ],
            dtype=float,
        )
        expected_3d = (
            center_plane["origin_center"]
            + expected_coords[:, :1] * center_plane["e1"]
            + expected_coords[:, 1:2] * center_plane["e2"]
        )
        image_points = _project(K, expected_3d)

        coords, lifted_3d = lift_points_to_center_plane(image_points, center_plane)
        reprojected = _project(K, lifted_3d)

        self.assertTrue(np.allclose(coords, expected_coords, atol=1e-8))
        self.assertTrue(np.allclose(lifted_3d, expected_3d, atol=1e-8))
        self.assertTrue(np.allclose(reprojected, image_points, atol=1e-8))


if __name__ == "__main__":
    unittest.main()
