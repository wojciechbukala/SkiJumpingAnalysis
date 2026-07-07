from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class TrajectoryConfig:
    video_path: Path
    csv_path: Path
    reference_frame: int
    selected_camera_id: int
    warp_mode: int
    allow_affine_fallback: bool
    min_valid_motion_score: float
    min_reliable_cumulative_score: float
    viewer_backend: str
    open_viewer: bool
    save_static_plot: bool
    plot_only_reliable_points: bool
    camera_intervals_json_path: Path
    reference_points_csv_path: Path
    trajectory_png_path: Path
    line_width: float
    point_size: float
    trajectory_color: tuple[float, float, float]
    ruling_color: tuple[float, float, float]


def default_trajectory_config() -> TrajectoryConfig:
    output_dir = Path("outputGeometry")
    return TrajectoryConfig(
        video_path=Path("detectionOutputs/G_30out.mp4"),
        csv_path=Path("detectionOutputs/G_30trajectory.csv"),
        reference_frame=157,
        selected_camera_id=1,
        warp_mode=cv2.MOTION_HOMOGRAPHY,
        allow_affine_fallback=True,
        min_valid_motion_score=1e-6,
        min_reliable_cumulative_score=0.30,
        viewer_backend="matplotlib",
        open_viewer=True,
        save_static_plot=True,
        plot_only_reliable_points=True,
        camera_intervals_json_path=output_dir / "jumper_trajectory_camera_intervals.json",
        reference_points_csv_path=output_dir / "jumper_trajectory_reference_points.csv",
        trajectory_png_path=output_dir / "jumper_trajectory_3d.png",
        line_width=2.0,
        point_size=12.0,
        trajectory_color=(1.0, 0.55, 0.0),
        ruling_color=(0.68, 0.68, 0.68),
    )
