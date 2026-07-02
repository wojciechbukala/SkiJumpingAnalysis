from __future__ import annotations

import unittest

import numpy as np

from src.track_geometry.models import (
    CameraCalibration,
    ImageCorrespondences,
    ImageCurve,
    ParallelCurveObservation,
)
from src.track_geometry.reconstruction import reconstruct_parallel_curves


def _project(K: np.ndarray, points_3d: np.ndarray) -> np.ndarray:
    projected = (K @ points_3d.T).T
    return projected[:, :2] / projected[:, 2:3]


class ParallelCurveReconstructionTest(unittest.TestCase):
    def test_metric_reconstruction_from_synthetic_correspondences(self) -> None:
        K = np.array(
            [
                [1000.0, 0.0, 960.0],
                [0.0, 1000.0, 540.0],
                [0.0, 0.0, 1.0],
            ],
        )
        normal = np.array([0.38, 0.0, 0.924986486])
        normal = normal / np.linalg.norm(normal)
        tangent = np.array([normal[2], 0.0, -normal[0]])
        vertical = np.array([0.0, 1.0, 0.0])
        origin = 18.0 * normal
        width = 4.0

        left_3d = []
        for s in np.linspace(-7.0, 13.0, 12):
            y = -0.025 * (s - 1.0) ** 2 + 2.0
            left_3d.append(origin + s * tangent + y * vertical)
        left_3d = np.vstack(left_3d)
        right_3d = left_3d + width * normal

        left_img = _project(K, left_3d)
        right_img = _project(K, right_3d)
        observation = ParallelCurveObservation(
            left_curve=ImageCurve(left_img, name="left"),
            right_curve=ImageCurve(right_img, name="right"),
            correspondences=ImageCorrespondences(left_img, right_img),
            camera=CameraCalibration(K),
            track_width=width,
        )

        result = reconstruct_parallel_curves(observation)

        self.assertTrue(result.has_metric_reconstruction)
        self.assertGreaterEqual(result.quality.positive_depth_ratio, 0.99)
        self.assertLess(result.quality.mean_ruling_residual, 1e-8)
        self.assertLess(result.quality.max_plane_error, 1e-8)
        reconstructed_widths = (result.right_3d - result.left_3d) @ result.plane_normal
        self.assertTrue(np.allclose(reconstructed_widths, width, atol=1e-8))


if __name__ == "__main__":
    unittest.main()
