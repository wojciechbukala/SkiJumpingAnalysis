from __future__ import annotations

from pathlib import Path

import cv2

from src.jumperTrajectory.config import TrajectoryConfig
from src.jumperTrajectory.motion import (
    detect_camera_intervals,
    normalize_homography,
    read_gray_frame_at,
    selected_interval,
)
from src.jumperTrajectory.motion import estimate_pairwise_homographies as _estimate_pairwise_homographies
from src.jumperTrajectory.outputs import (
    image_size_from_video,
    interval_to_dict,
)
from src.jumperTrajectory.outputs import save_camera_intervals as _save_camera_intervals
from src.jumperTrajectory.outputs import save_reference_points_csv as _save_reference_points_csv
from src.jumperTrajectory.plane import build_center_plane_homography, lift_points_to_center_plane
from src.jumperTrajectory.stabilization import (
    apply_homography_to_points,
    compute_cumulative_motion_scores,
    compute_frame_to_reference_transforms,
    read_kalman_points,
    stabilize_kalman_points,
)
from src.jumperTrajectory.viewer import (
    camera_open3d_geometry,
    draw_camera_matplotlib,
    open3d_line_set,
)
from src.jumperTrajectory.viewer import draw_scene_matplotlib as _draw_scene_matplotlib
from src.jumperTrajectory.viewer import maybe_show_viewer as _maybe_show_viewer
from src.jumperTrajectory.viewer import plot_static_3d as _plot_static_3d
from src.jumperTrajectory.viewer import show_matplotlib_viewer as _show_matplotlib_viewer
from src.jumperTrajectory.viewer import show_open3d_viewer as _show_open3d_viewer
from src.jumperTrajectory.viewer import show_pyvista_viewer as _show_pyvista_viewer
from src.jumperTrajectory.workflow import run_trajectory_pipeline


VIDEO_PATH = Path("detectionOutputs/G_30out.mp4")
CSV_PATH = Path("detectionOutputs/G_30trajectory.csv")
REFERENCE_FRAME = 157
SELECTED_CAMERA_ID = 1
WARP_MODE = cv2.MOTION_HOMOGRAPHY
ALLOW_AFFINE_FALLBACK = True
MIN_VALID_MOTION_SCORE = 1e-6
MIN_RELIABLE_CUMULATIVE_SCORE = 0.30
VIEWER_BACKEND = "matplotlib"
OPEN_VIEWER = True
SAVE_STATIC_PLOT = True
PLOT_ONLY_RELIABLE_POINTS = True

OUTPUT_DIR = Path("outputGeometry")
CAMERA_INTERVALS_JSON_PATH = OUTPUT_DIR / "jumper_trajectory_camera_intervals.json"
REFERENCE_POINTS_CSV_PATH = OUTPUT_DIR / "jumper_trajectory_reference_points.csv"
TRAJECTORY_PNG_PATH = OUTPUT_DIR / "jumper_trajectory_3d.png"

LINE_WIDTH = 2.0
POINT_SIZE = 12.0
TRAJECTORY_COLOR = (1.0, 0.55, 0.0)
RULING_COLOR = (0.68, 0.68, 0.68)


def build_config() -> TrajectoryConfig:
    return TrajectoryConfig(
        video_path=VIDEO_PATH,
        csv_path=CSV_PATH,
        reference_frame=REFERENCE_FRAME,
        selected_camera_id=SELECTED_CAMERA_ID,
        warp_mode=WARP_MODE,
        allow_affine_fallback=ALLOW_AFFINE_FALLBACK,
        min_valid_motion_score=MIN_VALID_MOTION_SCORE,
        min_reliable_cumulative_score=MIN_RELIABLE_CUMULATIVE_SCORE,
        viewer_backend=VIEWER_BACKEND,
        open_viewer=OPEN_VIEWER,
        save_static_plot=SAVE_STATIC_PLOT,
        plot_only_reliable_points=PLOT_ONLY_RELIABLE_POINTS,
        camera_intervals_json_path=CAMERA_INTERVALS_JSON_PATH,
        reference_points_csv_path=REFERENCE_POINTS_CSV_PATH,
        trajectory_png_path=TRAJECTORY_PNG_PATH,
        line_width=LINE_WIDTH,
        point_size=POINT_SIZE,
        trajectory_color=TRAJECTORY_COLOR,
        ruling_color=RULING_COLOR,
    )


def estimate_pairwise_homographies(video_path, provider, start_frame, end_frame, warp_mode):
    return _estimate_pairwise_homographies(
        video_path,
        provider,
        start_frame,
        end_frame,
        warp_mode,
        ALLOW_AFFINE_FALLBACK,
        MIN_VALID_MOTION_SCORE,
    )


def save_camera_intervals(path, video_path, changes, intervals, fps, total_frames) -> None:
    _save_camera_intervals(
        path,
        video_path,
        changes,
        intervals,
        fps,
        total_frames,
        REFERENCE_FRAME,
        SELECTED_CAMERA_ID,
    )


def save_reference_points_csv(
    path,
    frames,
    image_points,
    reference_points,
    plane_coords,
    trajectory_3d,
    motion_scores,
    fps,
) -> None:
    _save_reference_points_csv(
        path,
        frames,
        image_points,
        reference_points,
        plane_coords,
        trajectory_3d,
        motion_scores,
        fps,
        SELECTED_CAMERA_ID,
        MIN_RELIABLE_CUMULATIVE_SCORE,
    )


def plot_static_3d(path, slope, trajectory_3d, frames, K, image_size) -> None:
    _plot_static_3d(path, slope, trajectory_3d, frames, K, image_size, build_config())


def draw_scene_matplotlib(ax, slope, trajectory_3d, frames, K, image_size) -> None:
    _draw_scene_matplotlib(ax, slope, trajectory_3d, frames, K, image_size, build_config())


def show_matplotlib_viewer(slope, trajectory_3d, frames, K, image_size) -> None:
    _show_matplotlib_viewer(slope, trajectory_3d, frames, K, image_size, build_config())


def show_open3d_viewer(slope, trajectory_3d, K, image_size) -> None:
    _show_open3d_viewer(slope, trajectory_3d, K, image_size, build_config())


def show_pyvista_viewer(slope, trajectory_3d, frames, K, image_size) -> None:
    _show_pyvista_viewer(slope, trajectory_3d, frames, K, image_size, build_config())


def maybe_show_viewer(slope, trajectory_3d, frames, K, image_size) -> None:
    _maybe_show_viewer(slope, trajectory_3d, frames, K, image_size, build_config())


def run_pipeline() -> dict:
    return run_trajectory_pipeline(build_config())


def main() -> int:
    run_pipeline()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
