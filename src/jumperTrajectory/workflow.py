from __future__ import annotations

import os

import numpy as np

from src.SwitchCam import switchCam
from src.jumperTrajectory.config import TrajectoryConfig
from src.jumperTrajectory.motion import (
    detect_camera_intervals,
    estimate_pairwise_homographies,
    selected_interval,
)
from src.jumperTrajectory.outputs import (
    image_size_from_video,
    save_camera_intervals,
    save_reference_points_csv,
)
from src.jumperTrajectory.plane import build_center_plane_homography, lift_points_to_center_plane
from src.jumperTrajectory.stabilization import (
    compute_cumulative_motion_scores,
    compute_frame_to_reference_transforms,
    read_kalman_points,
    stabilize_kalman_points,
)
from src.jumperTrajectory.viewer import maybe_show_viewer, plot_static_3d
from viz_slope import (
    camera_matrix_from_annotations,
    load_annotations,
    load_homography_bundle,
    load_homography_corrected_reconstruction,
    load_reconstruction,
)


os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/skijumping-matplotlib")


def run_trajectory_pipeline(config: TrajectoryConfig) -> dict:
    if not config.video_path.exists():
        raise FileNotFoundError(f"Video non trovato: {config.video_path}")
    if not config.csv_path.exists():
        raise FileNotFoundError(f"CSV non trovato: {config.csv_path}")

    changes, intervals, fps, total_frames = detect_camera_intervals(config.video_path)
    save_camera_intervals(
        config.camera_intervals_json_path,
        config.video_path,
        changes,
        intervals,
        fps,
        total_frames,
        config.reference_frame,
        config.selected_camera_id,
    )
    interval = selected_interval(intervals, config.selected_camera_id, config.reference_frame)
    start_frame = interval.start.frame_idx
    end_frame = interval.end.frame_idx

    print(f"Video: {config.video_path}", flush=True)
    print(f"FPS: {fps:.3f}, frame totali: {total_frames}", flush=True)
    print(switchCam.format_camera_intervals(intervals), flush=True)
    print(f"Uso camera {config.selected_camera_id}: frame {start_frame}-{end_frame}", flush=True)
    print(f"Frame riferimento: {config.reference_frame}", flush=True)

    from src.Mask.Masking import JumperProvider

    provider = JumperProvider(str(config.csv_path))
    kalman_points = read_kalman_points(config.csv_path, start_frame, end_frame)
    pairwise_warps, motion_rows = estimate_pairwise_homographies(
        config.video_path,
        provider,
        start_frame,
        end_frame,
        config.warp_mode,
        config.allow_affine_fallback,
        config.min_valid_motion_score,
    )
    transforms = compute_frame_to_reference_transforms(
        pairwise_warps,
        start_frame,
        end_frame,
        config.reference_frame,
    )
    cumulative_scores = compute_cumulative_motion_scores(
        motion_rows,
        start_frame,
        end_frame,
        config.reference_frame,
    )
    frames, image_points, reference_points = stabilize_kalman_points(kalman_points, transforms)
    motion_scores = np.asarray([cumulative_scores.get(int(frame), 0.0) for frame in frames], dtype=float)

    bundle = load_homography_bundle()
    if "K" in bundle:
        K = np.asarray(bundle["K"], dtype=float)
    else:
        annotations = load_annotations()
        K = camera_matrix_from_annotations(annotations)
    slope_raw = load_reconstruction()
    slope = load_homography_corrected_reconstruction(slope_raw)

    center_plane = build_center_plane_homography(bundle, K)
    plane_coords, trajectory_3d = lift_points_to_center_plane(reference_points, center_plane)
    image_size = image_size_from_video(config.video_path)
    if config.plot_only_reliable_points:
        plot_mask = motion_scores >= config.min_reliable_cumulative_score
    else:
        plot_mask = np.ones(len(frames), dtype=bool)
    if not np.any(plot_mask):
        plot_mask = np.ones(len(frames), dtype=bool)
    plot_frames = frames[plot_mask]
    plot_trajectory_3d = trajectory_3d[plot_mask]

    save_reference_points_csv(
        config.reference_points_csv_path,
        frames,
        image_points,
        reference_points,
        plane_coords,
        trajectory_3d,
        motion_scores,
        fps,
        config.selected_camera_id,
        config.min_reliable_cumulative_score,
    )
    if config.save_static_plot:
        plot_static_3d(config.trajectory_png_path, slope, plot_trajectory_3d, plot_frames, K, image_size, config)

    print(f"Punti traiettoria validi: {len(frames)}", flush=True)
    reliable_count = int(np.sum(motion_scores >= config.min_reliable_cumulative_score))
    print(f"Punti affidabili score>={config.min_reliable_cumulative_score}: {reliable_count}", flush=True)
    print(f"Frame iniziale/finale traiettoria: {int(frames[0])}-{int(frames[-1])}", flush=True)
    print(f"Output intervalli: {config.camera_intervals_json_path}", flush=True)
    print(f"Output punti: {config.reference_points_csv_path}", flush=True)
    if config.save_static_plot:
        print(f"Plot 3D: {config.trajectory_png_path}", flush=True)

    maybe_show_viewer(slope, plot_trajectory_3d, plot_frames, K, image_size, config)
    return {
        "frames": frames,
        "image_points": image_points,
        "reference_points": reference_points,
        "plane_coords": plane_coords,
        "trajectory_3d": trajectory_3d,
        "motion_scores": motion_scores,
        "interval": interval,
        "fps": fps,
        "K": K,
        "center_plane": center_plane,
    }
