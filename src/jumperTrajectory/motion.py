from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from src.SwitchCam import switchCam

if TYPE_CHECKING:
    from src.Mask.Masking import JumperProvider


def normalize_homography(matrix: np.ndarray) -> np.ndarray:
    H = np.asarray(matrix, dtype=float)
    if H.shape == (2, 3):
        H = np.vstack([H, [0.0, 0.0, 1.0]])
    if H.shape != (3, 3):
        raise ValueError(f"Omografia non valida, shape={H.shape}.")
    if abs(float(H[2, 2])) > 1e-12:
        return H / H[2, 2]
    norm = float(np.linalg.norm(H))
    if norm < 1e-12:
        raise ValueError("Omografia degenere con norma quasi nulla.")
    return H / norm


def detect_camera_intervals(video_path: Path) -> tuple[list[switchCam.ShotChange], list[switchCam.CameraInterval], float, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire il video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        cap.release()
        raise RuntimeError(f"FPS non valido: {fps}")
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Numero frame non valido: {total_frames}")

    changes = switchCam.detect_shot_changes(cap, fps=fps)
    cap.release()
    intervals = switchCam.get_camera_intervals(changes, total_frames, fps)
    return changes, intervals, fps, total_frames


def selected_interval(
    intervals: list[switchCam.CameraInterval],
    camera_id: int,
    reference_frame: int,
) -> switchCam.CameraInterval:
    matches = [interval for interval in intervals if interval.camera_id == camera_id]
    if not matches:
        available = [interval.camera_id for interval in intervals]
        raise ValueError(f"Camera {camera_id} non trovata. Camere disponibili: {available}")

    interval = matches[0]
    if not interval.start.frame_idx <= reference_frame <= interval.end.frame_idx:
        raise ValueError(
            f"REFERENCE_FRAME={reference_frame} non appartiene alla camera {camera_id}: "
            f"intervallo {interval.start.frame_idx}-{interval.end.frame_idx}.",
        )
    return interval


def read_gray_frame_at(cap: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Impossibile leggere il frame {frame_index}.")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def estimate_pairwise_homographies(
    video_path: Path,
    provider: "JumperProvider",
    start_frame: int,
    end_frame: int,
    warp_mode: int,
    allow_affine_fallback: bool,
    min_valid_motion_score: float,
) -> tuple[dict[int, np.ndarray], list[dict]]:
    from src.transform import transform as motion_transform

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire il video: {video_path}")

    previous_idx = start_frame
    previous = read_gray_frame_at(cap, previous_idx)
    warps: dict[int, np.ndarray] = {}
    motion_rows: list[dict] = []

    for current_idx in range(start_frame + 1, end_frame + 1):
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"Impossibile leggere il frame {current_idx}.")
        current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score, warp = motion_transform.ecc_2frame(
            previous_idx,
            current_idx,
            previous,
            current,
            provider,
            warp_mode=warp_mode,
        )
        mode_label = "homography" if warp_mode == cv2.MOTION_HOMOGRAPHY else "affine"
        if allow_affine_fallback and warp_mode == cv2.MOTION_HOMOGRAPHY and score <= min_valid_motion_score:
            fallback_score, fallback_warp = motion_transform.ecc_2frame(
                previous_idx,
                current_idx,
                previous,
                current,
                provider,
                warp_mode=cv2.MOTION_AFFINE,
            )
            if fallback_score > score:
                score = fallback_score
                warp = fallback_warp
                mode_label = "affine_fallback"
        if score <= min_valid_motion_score:
            mode_label = "identity"
        warps[current_idx] = normalize_homography(warp)
        motion_rows.append(
            {
                "frame": current_idx,
                "previous_frame": previous_idx,
                "mode": mode_label,
                "score": float(score),
            },
        )
        print(f"Motion {current_idx}->{previous_idx}: mode={mode_label}, score={score:.4f}", flush=True)
        previous_idx = current_idx
        previous = current

    cap.release()
    return warps, motion_rows
