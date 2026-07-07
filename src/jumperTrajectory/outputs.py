from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2

from src.SwitchCam import switchCam


def interval_to_dict(interval: switchCam.CameraInterval) -> dict:
    return {
        "camera_id": interval.camera_id,
        "camera_type": interval.camera_type,
        "start_frame": interval.start.frame_idx,
        "end_frame": interval.end.frame_idx,
        "start_time_s": interval.start.time_s,
        "end_time_s": interval.end.time_s,
    }


def save_camera_intervals(
    path: Path,
    video_path: Path,
    changes: list[switchCam.ShotChange],
    intervals: list[switchCam.CameraInterval],
    fps: float,
    total_frames: int,
    reference_frame: int,
    selected_camera_id: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "video_path": str(video_path),
        "fps": fps,
        "total_frames": total_frames,
        "reference_frame": reference_frame,
        "selected_camera_id": selected_camera_id,
        "changes": [
            {"frame_idx": change.frame_idx, "time_s": change.time_s, "score": change.score}
            for change in changes
        ],
        "intervals": [interval_to_dict(interval) for interval in intervals],
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def save_reference_points_csv(
    path: Path,
    frames: np.ndarray,
    image_points: np.ndarray,
    reference_points: np.ndarray,
    plane_coords: np.ndarray,
    trajectory_3d: np.ndarray,
    motion_scores: np.ndarray,
    fps: float,
    selected_camera_id: int,
    min_reliable_cumulative_score: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame",
                "time_s",
                "camera_id",
                "kf_x",
                "kf_y",
                "ref_x",
                "ref_y",
                "plane_u",
                "plane_v",
                "x",
                "y",
                "z",
                "motion_score_to_ref",
                "reliable",
            ],
        )
        for frame, image_point, ref_point, plane_coord, point_3d, motion_score in zip(
            frames,
            image_points,
            reference_points,
            plane_coords,
            trajectory_3d,
            motion_scores,
        ):
            writer.writerow(
                [
                    int(frame),
                    float(frame) / fps,
                    selected_camera_id,
                    float(image_point[0]),
                    float(image_point[1]),
                    float(ref_point[0]),
                    float(ref_point[1]),
                    float(plane_coord[0]),
                    float(plane_coord[1]),
                    float(point_3d[0]),
                    float(point_3d[1]),
                    float(point_3d[2]),
                    float(motion_score),
                    int(float(motion_score) >= min_reliable_cumulative_score),
                ],
            )


def image_size_from_video(video_path: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire il video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height
