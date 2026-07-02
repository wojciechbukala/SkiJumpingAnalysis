from __future__ import annotations

import unittest

import numpy as np

from slope_extraction import (
    build_vanishing_correspondences,
    estimate_vanishing_point_from_segments,
    reconstruct_curves_3d,
)
from src.track_geometry.models import CameraCalibration


def _project(K: np.ndarray, points_3d: np.ndarray) -> np.ndarray:
    projected = (K @ points_3d.T).T
    return projected[:, :2] / projected[:, 2:3]


class SlopeExtractionTest(unittest.TestCase):
    def test_vanishing_point_from_multiple_concurrent_segments(self) -> None:
        expected = np.array([1120.0, -240.0, 1.0], dtype=float)
        anchors = [
            np.array([120.0, 180.0], dtype=float),
            np.array([360.0, 640.0], dtype=float),
            np.array([760.0, 420.0], dtype=float),
            np.array([980.0, 880.0], dtype=float),
        ]
        segments = []
        for anchor in anchors:
            second = anchor + 0.28 * (expected[:2] - anchor)
            segments.append(((float(anchor[0]), float(anchor[1])), (float(second[0]), float(second[1]))))

        actual = estimate_vanishing_point_from_segments(segments)

        self.assertTrue(np.allclose(actual / actual[2], expected, atol=1e-8))

    def test_vanishing_correspondences_keep_order_and_choose_reversed_curve(self) -> None:
        vp = np.array([900.0, 80.0, 1.0], dtype=float)
        x = np.linspace(180.0, 620.0, 18)
        y = 410.0 + 0.0008 * (x - 400.0) ** 2 + 25.0 * np.sin(np.linspace(0.0, np.pi, len(x)))
        curve_a = np.column_stack([x, y])
        curve_b = vp[:2] + 1.18 * (curve_a - vp[:2])

        result = build_vanishing_correspondences(curve_a, curve_b[::-1], vp, num_samples=36)

        self.assertTrue(result.reversed_curve_b)
        self.assertLess(result.mean_error, 0.1)
        self.assertTrue(np.all(np.diff(result.points_a[:, 0]) > 0.0))
        self.assertTrue(np.all(np.diff(result.points_b[:, 0]) > 0.0))

    def test_reconstruction_recovers_relative_width_from_synthetic_curves(self) -> None:
        K = np.array(
            [
                [1250.0, 0.0, 960.0],
                [0.0, 970.0, 540.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        normal = np.array([0.34, -0.08, 0.936], dtype=float)
        normal = normal / np.linalg.norm(normal)
        tangent = np.array([normal[2], 0.0, -normal[0]], dtype=float)
        tangent = tangent / np.linalg.norm(tangent)
        vertical = np.cross(normal, tangent)
        vertical = vertical / np.linalg.norm(vertical)
        origin = np.array([-1.0, 0.4, 13.0], dtype=float)
        width = 1.75

        left_3d = []
        for value in np.linspace(-2.5, 3.5, 30):
            height = 0.14 * np.sin(value * 1.3) + 0.05 * value
            left_3d.append(origin + value * tangent + height * vertical)
        left_3d = np.vstack(left_3d)
        right_3d = left_3d + width * normal
        left_img = _project(K, left_3d)
        right_img = _project(K, right_3d)

        result = reconstruct_curves_3d(CameraCalibration(K), left_img, right_img, normal, width=width)
        reconstructed_widths = np.linalg.norm(result.right_3d - result.left_3d, axis=1)
        left_plane_offsets = result.left_3d @ result.normal
        right_plane_offsets = result.right_3d @ result.normal

        self.assertGreaterEqual(result.positive_depth_ratio, 0.99)
        self.assertLess(result.mean_residual, 1e-8)
        self.assertGreater(float(np.dot(result.normal, normal)), 0.99)
        self.assertTrue(np.allclose(reconstructed_widths, width, atol=1e-8))
        self.assertLess(float(np.std(left_plane_offsets)), 1e-8)
        self.assertLess(float(np.std(right_plane_offsets)), 1e-8)
        self.assertAlmostEqual(float(np.mean(right_plane_offsets - left_plane_offsets)), width, places=8)


if __name__ == "__main__":
    unittest.main()
