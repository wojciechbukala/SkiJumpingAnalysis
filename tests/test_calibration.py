from __future__ import annotations

import unittest

import numpy as np

from src.track_geometry.calibration import (
    estimate_k_from_orthogonal_vanishing_pairs,
    estimate_k_from_orthogonal_vanishing_pairs_and_equal_lengths,
)


class CalibrationTest(unittest.TestCase):
    def test_k_from_independent_focal_scales_with_known_principal_point(self) -> None:
        fx = 920.0
        fy = 710.0
        cx = 320.0
        cy = 240.0

        d1 = np.array([1.0 / np.sqrt(2.0), 0.0, 1.0 / np.sqrt(2.0)])
        d2 = np.array([-1.0 / np.sqrt(6.0), 2.0 / np.sqrt(6.0), 1.0 / np.sqrt(6.0)])
        d3 = np.cross(d1, d2)

        def project_direction(direction: np.ndarray) -> np.ndarray:
            return np.array(
                [
                    fx * direction[0] / direction[2] + cx,
                    fy * direction[1] / direction[2] + cy,
                ],
            )

        vx = project_direction(d1)
        vy = project_direction(d2)
        vz = project_direction(d3)

        pairs = [
            [vx, vy],
            [vx, vz],
            [vy, vz],
        ]

        K = estimate_k_from_orthogonal_vanishing_pairs(
            pairs,
            image_shape=(480, 640),
            principal_point=(cx, cy),
        )

        self.assertAlmostEqual(K[0, 0], fx, places=8)
        self.assertAlmostEqual(K[1, 1], fy, places=8)
        self.assertAlmostEqual(K[0, 2], cx, places=8)
        self.assertAlmostEqual(K[1, 2], cy, places=8)
        self.assertTrue(np.allclose(K[2], [0.0, 0.0, 1.0]))

    def test_k_from_three_orthogonal_pairs_and_equal_lengths(self) -> None:
        fx = 910.0
        fy = 760.0
        cx = 330.0
        cy = 245.0
        K_true = np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

        raw_basis = np.array(
            [
                [0.7, -0.2, 0.5],
                [0.3, 0.9, -0.4],
                [0.65, 0.35, 0.75],
            ],
            dtype=float,
        )
        basis, _ = np.linalg.qr(raw_basis)
        for idx in range(3):
            if basis[2, idx] < 0.0:
                basis[:, idx] *= -1.0

        def project_direction(direction: np.ndarray) -> np.ndarray:
            projected = K_true @ direction
            return projected[:2] / projected[2]

        vanishing_points = [project_direction(basis[:, idx]) for idx in range(3)]
        pairs = [
            [vanishing_points[0], vanishing_points[1]],
            [vanishing_points[0], vanishing_points[2]],
            [vanishing_points[1], vanishing_points[2]],
        ]

        plane_origin = 12.0 * basis[:, 2]

        def project_plane_point(u_coord: float, v_coord: float) -> np.ndarray:
            point_3d = plane_origin + u_coord * basis[:, 0] + v_coord * basis[:, 1]
            projected = K_true @ point_3d
            return projected[:2] / projected[2]

        equal_length_segments = [
            [project_plane_point(0.0, 0.0), project_plane_point(1.5, 0.3)],
            [project_plane_point(2.0, 1.0), project_plane_point(2.3, 2.5)],
        ]

        K = estimate_k_from_orthogonal_vanishing_pairs_and_equal_lengths(
            pairs,
            equal_length_segments,
            vanishing_points[2],
            image_shape=(480, 640),
        )

        self.assertAlmostEqual(K[0, 0], fx, places=6)
        self.assertAlmostEqual(K[1, 1], fy, places=6)
        self.assertAlmostEqual(K[0, 2], cx, places=6)
        self.assertAlmostEqual(K[1, 2], cy, places=6)
        self.assertTrue(np.allclose(K[2], [0.0, 0.0, 1.0]))


if __name__ == "__main__":
    unittest.main()
