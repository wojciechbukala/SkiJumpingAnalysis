# Jumper trajectory reconstruction workflow
# Step one detects shot changes and camera intervals in the video
# Step two selects the configured camera interval containing REFERENCE_FRAME
# Step three reads Kalman jumper points from the detection CSV inside that interval
# Step four estimates consecutive frame motion with SIFT + RANSAC homographies and optional affine fallback
# Step five composes frame to reference homographies forward and backward from REFERENCE_FRAME
# Step six warps tracked jumper points into the reference image
# Step seven loads saved slope geometry from viz_slope and reads the center plane homography
# Step eight lifts stabilized image points to the center plane and saves CSV NPZ and optional plot output
# Step nine displays the trajectory with Open3D Matplotlib or automatic fallback
# Option SELECTED_CAMERA_ID chooses the video interval to analyze
# Option WARP_MODE chooses homography or affine SIFT RANSAC motion estimation
# Option ALLOW_AFFINE_FALLBACK permits affine fallback when homography score is poor
# Option MIN_RELIABLE_CUMULATIVE_SCORE filters low confidence trajectory points for plotting
# Option VIEWER_BACKEND chooses auto open3d or matplotlib
# Option OPEN_VIEWER SAVE_STATIC_PLOT and PLOT_ONLY_RELIABLE_POINTS control outputs and display
# Formula normalized homography is H divided by H[2,2] when possible
# Formula point warp is p_ref = H @ [x y 1] followed by division by z
# Formula cumulative motion score is the minimum pairwise score along the path to the reference frame
# Formula center plane homography is read from viz_slope or built as H = K @ [e1 e2 origin_center]
# Formula lifted point is P = origin_center + u e1 + v e2

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import cv2
import numpy as np

from src.Mask.Masking import JumperProvider
from src.SwitchCam import switchCam
from src.transform import transform as motion_transform
from viz_slope import (
    BACKGROUND_COLOR,
    CAMERA_COLOR,
    CAMERA_X_COLOR,
    CAMERA_Y_COLOR,
    CAMERA_Z_COLOR,
    CENTER_COLOR,
    LEFT_COLOR,
    RIGHT_COLOR,
    build_plane_homography,
    camera_axis_points,
    camera_frustum_points,
    camera_matrix_from_annotations,
    load_homography_bundle,
    load_homography_corrected_reconstruction,
    load_annotations,
    load_reconstruction,
    set_axes_equal,
    unit,
)


VIDEO_PATH = Path("detectionOutputs/G_30out.mp4")
CSV_PATH = Path("detectionOutputs/G_30trajectory.csv")
REFERENCE_FRAME = 157
SELECTED_CAMERA_ID = 1
WARP_MODE = cv2.MOTION_HOMOGRAPHY
ALLOW_AFFINE_FALLBACK = True
MIN_VALID_MOTION_SCORE = 1e-6
MIN_RELIABLE_CUMULATIVE_SCORE = 0.30
VIEWER_BACKEND = "auto"
OPEN_VIEWER = True
SAVE_STATIC_PLOT = True
PLOT_ONLY_RELIABLE_POINTS = True

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/skijumping-matplotlib")

OUTPUT_DIR = Path("outputGeometry")
CAMERA_INTERVALS_JSON_PATH = OUTPUT_DIR / "jumper_trajectory_camera_intervals.json"
REFERENCE_POINTS_CSV_PATH = OUTPUT_DIR / "jumper_trajectory_reference_points.csv"
TRAJECTORY_NPZ_PATH = OUTPUT_DIR / "jumper_trajectory_3d.npz"
TRAJECTORY_PNG_PATH = OUTPUT_DIR / "jumper_trajectory_3d.png"

LINE_WIDTH = 2.0
POINT_SIZE = 12.0
TRAJECTORY_COLOR = (1.0, 0.55, 0.0)
RULING_COLOR = (0.68, 0.68, 0.68)
CAMERA_FRUSTUM_SCALE = 0.28
CAMERA_AXIS_SCALE = 0.45


# Normalize a homography matrix
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


# Transform points with a homography
def apply_homography_to_points(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    points_h = np.column_stack([points, np.ones(len(points), dtype=float)])
    mapped = (H @ points_h.T).T
    valid = np.abs(mapped[:, 2]) > 1e-12
    output = np.full((len(points), 2), np.nan, dtype=float)
    output[valid] = mapped[valid, :2] / mapped[valid, 2:3]
    return output


# Detect camera intervals in video
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


# Convert an interval to data
def interval_to_dict(interval: switchCam.CameraInterval) -> dict:
    return {
        "camera_id": interval.camera_id,
        "camera_type": interval.camera_type,
        "start_frame": interval.start.frame_idx,
        "end_frame": interval.end.frame_idx,
        "start_time_s": interval.start.time_s,
        "end_time_s": interval.end.time_s,
    }


# Save camera interval data
def save_camera_intervals(
    path: Path,
    video_path: Path,
    changes: list[switchCam.ShotChange],
    intervals: list[switchCam.CameraInterval],
    fps: float,
    total_frames: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "video_path": str(video_path),
        "fps": fps,
        "total_frames": total_frames,
        "reference_frame": REFERENCE_FRAME,
        "selected_camera_id": SELECTED_CAMERA_ID,
        "changes": [
            {"frame_idx": change.frame_idx, "time_s": change.time_s, "score": change.score}
            for change in changes
        ],
        "intervals": [interval_to_dict(interval) for interval in intervals],
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


# Select an interval for analysis
def selected_interval(intervals: list[switchCam.CameraInterval], camera_id: int, reference_frame: int) -> switchCam.CameraInterval:
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


# Read tracked jumper points
def read_kalman_points(csv_path: Path, start_frame: int, end_frame: int) -> dict[int, tuple[float, float]]:
    points: dict[int, tuple[float, float]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"frame", "kf_x", "kf_y"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV incompleto, mancano colonne: {sorted(missing)}")

        for row in reader:
            frame = int(float(row["frame"]))
            if frame < start_frame or frame > end_frame:
                continue
            x = float(row["kf_x"])
            y = float(row["kf_y"])
            if np.isfinite(x) and np.isfinite(y):
                points[frame] = (x, y)

    if not points:
        raise ValueError(f"Nessun punto Kalman valido nel range {start_frame}-{end_frame}.")
    return points


# Read one grayscale frame
def read_gray_frame_at(cap: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Impossibile leggere il frame {frame_index}.")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


# Estimate motion between frames
def estimate_pairwise_homographies(
    video_path: Path,
    provider: JumperProvider,
    start_frame: int,
    end_frame: int,
    warp_mode: int,
) -> dict[int, np.ndarray]:
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
        if (
            ALLOW_AFFINE_FALLBACK
            and warp_mode == cv2.MOTION_HOMOGRAPHY
            and score <= MIN_VALID_MOTION_SCORE
        ):
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
        if score <= MIN_VALID_MOTION_SCORE:
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


# Compute transforms to reference frame
def compute_frame_to_reference_transforms(
    pairwise_warps: dict[int, np.ndarray],
    start_frame: int,
    end_frame: int,
    reference_frame: int,
) -> dict[int, np.ndarray]:
    if not start_frame <= reference_frame <= end_frame:
        raise ValueError("Il frame di riferimento deve stare dentro l'intervallo camera.")

    transforms: dict[int, np.ndarray] = {reference_frame: np.eye(3, dtype=float)}

    for frame in range(reference_frame + 1, end_frame + 1):
        if frame not in pairwise_warps:
            raise ValueError(f"Manca omografia consecutiva {frame}->{frame - 1}.")
        transforms[frame] = normalize_homography(transforms[frame - 1] @ pairwise_warps[frame])

    for frame in range(reference_frame - 1, start_frame - 1, -1):
        next_frame = frame + 1
        if next_frame not in pairwise_warps:
            raise ValueError(f"Manca omografia consecutiva {next_frame}->{frame}.")
        transforms[frame] = normalize_homography(transforms[next_frame] @ np.linalg.inv(pairwise_warps[next_frame]))

    return transforms


# Compute accumulated motion scores
def compute_cumulative_motion_scores(
    motion_rows: list[dict],
    start_frame: int,
    end_frame: int,
    reference_frame: int,
) -> dict[int, float]:
    pair_scores = {int(row["frame"]): float(row["score"]) for row in motion_rows}
    scores = {reference_frame: 1.0}

    for frame in range(reference_frame + 1, end_frame + 1):
        scores[frame] = min(scores[frame - 1], pair_scores.get(frame, 0.0))

    for frame in range(reference_frame - 1, start_frame - 1, -1):
        scores[frame] = min(scores[frame + 1], pair_scores.get(frame + 1, 0.0))

    return scores


# Stabilize tracked jumper points
def stabilize_kalman_points(
    kalman_points: dict[int, tuple[float, float]],
    transforms: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = np.array(sorted(frame for frame in kalman_points if frame in transforms), dtype=int)
    if len(frames) == 0:
        raise ValueError("Nessun frame ha sia punto Kalman sia omografia verso il riferimento.")

    image_points = np.array([kalman_points[int(frame)] for frame in frames], dtype=float)
    reference_points = np.vstack(
        [
            apply_homography_to_points(transforms[int(frame)], image_points[idx : idx + 1])[0]
            for idx, frame in enumerate(frames)
        ],
    )
    valid = np.all(np.isfinite(reference_points), axis=1)
    return frames[valid], image_points[valid], reference_points[valid]


# Build the center plane mapping
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


# Lift image points to the center plane
def lift_points_to_center_plane(reference_points: np.ndarray, center_plane: dict) -> tuple[np.ndarray, np.ndarray]:
    coords = apply_homography_to_points(center_plane["H_image_to_center_plane"], reference_points)
    points_3d = (
        center_plane["origin_center"]
        + coords[:, :1] * center_plane["e1"]
        + coords[:, 1:2] * center_plane["e2"]
    )
    return coords, points_3d


# Save stabilized reference points
def save_reference_points_csv(
    path: Path,
    frames: np.ndarray,
    image_points: np.ndarray,
    reference_points: np.ndarray,
    plane_coords: np.ndarray,
    trajectory_3d: np.ndarray,
    motion_scores: np.ndarray,
    fps: float,
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
                    SELECTED_CAMERA_ID,
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
                    int(float(motion_score) >= MIN_RELIABLE_CUMULATIVE_SCORE),
                ],
            )


# Save reconstructed trajectory data
def save_trajectory_npz(
    path: Path,
    frames: np.ndarray,
    image_points: np.ndarray,
    reference_points: np.ndarray,
    plane_coords: np.ndarray,
    trajectory_3d: np.ndarray,
    motion_scores: np.ndarray,
    center_plane: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        frames=frames,
        image_points=image_points,
        reference_points=reference_points,
        plane_coords=plane_coords,
        trajectory_3d=trajectory_3d,
        motion_scores=motion_scores,
        reliable_mask=motion_scores >= MIN_RELIABLE_CUMULATIVE_SCORE,
        origin_center=center_plane["origin_center"],
        e1=center_plane["e1"],
        e2=center_plane["e2"],
        normal=center_plane["normal"],
        H_center_plane_to_image=center_plane["H_center_plane_to_image"],
        H_image_to_center_plane=center_plane["H_image_to_center_plane"],
        reference_frame=np.asarray([REFERENCE_FRAME], dtype=int),
        selected_camera_id=np.asarray([SELECTED_CAMERA_ID], dtype=int),
    )


# Draw the camera in a plot
def draw_camera_matplotlib(ax, K: np.ndarray, image_size: tuple[int, int], scene_points: np.ndarray) -> np.ndarray:
    corners = camera_frustum_points(K, image_size, scene_points)
    axis_points = camera_axis_points(scene_points)
    origin = np.zeros(3, dtype=float)
    for corner in corners:
        segment = np.vstack([origin, corner])
        ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color=CAMERA_COLOR, linewidth=1.0)
    for first, second in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        segment = np.vstack([corners[first], corners[second]])
        ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color=CAMERA_COLOR, linewidth=1.0)
    ax.scatter([0.0], [0.0], [0.0], color=CAMERA_COLOR, s=30, label="camera")
    ax.quiver(0.0, 0.0, 0.0, axis_points[1, 0], axis_points[1, 1], axis_points[1, 2], color=CAMERA_X_COLOR)
    ax.quiver(0.0, 0.0, 0.0, axis_points[2, 0], axis_points[2, 1], axis_points[2, 2], color=CAMERA_Y_COLOR)
    ax.quiver(0.0, 0.0, 0.0, axis_points[3, 0], axis_points[3, 1], axis_points[3, 2], color=CAMERA_Z_COLOR)
    return np.vstack([origin, corners, axis_points])


# Read image dimensions from video
def image_size_from_video(video_path: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire il video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height


# Save a static trajectory plot
def plot_static_3d(
    path: Path,
    slope: dict,
    trajectory_3d: np.ndarray,
    frames: np.ndarray,
    K: np.ndarray,
    image_size: tuple[int, int],
) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    path.parent.mkdir(parents=True, exist_ok=True)
    fig = Figure(figsize=(11, 8))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection="3d")
    draw_scene_matplotlib(ax, slope, trajectory_3d, frames, K, image_size)
    fig.tight_layout()
    fig.savefig(path, dpi=160)


# Draw the trajectory scene
def draw_scene_matplotlib(
    ax,
    slope: dict,
    trajectory_3d: np.ndarray,
    frames: np.ndarray,
    K: np.ndarray,
    image_size: tuple[int, int],
) -> None:
    left = slope["left_3d"]
    right = slope["right_3d"]
    center = slope["center_3d"]
    all_points = np.vstack([left, right, center, trajectory_3d])

    ax.plot(left[:, 0], left[:, 1], left[:, 2], color=LEFT_COLOR, linewidth=LINE_WIDTH, label="corsia A")
    ax.plot(right[:, 0], right[:, 1], right[:, 2], color=RIGHT_COLOR, linewidth=LINE_WIDTH, label="corsia B")
    ax.plot(center[:, 0], center[:, 1], center[:, 2], "--", color=CENTER_COLOR, linewidth=1.5, label="linea centrale")

    stride = max(1, len(left) // 25)
    for idx in range(0, len(left), stride):
        segment = np.vstack([left[idx], right[idx]])
        ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color=RULING_COLOR, linewidth=0.8, alpha=0.7)

    ax.plot(
        trajectory_3d[:, 0],
        trajectory_3d[:, 1],
        trajectory_3d[:, 2],
        color=TRAJECTORY_COLOR,
        linewidth=2.5,
        label="traiettoria sciatore",
    )
    values = np.linspace(0.0, 1.0, len(trajectory_3d))
    scatter = ax.scatter(
        trajectory_3d[:, 0],
        trajectory_3d[:, 1],
        trajectory_3d[:, 2],
        c=values,
        cmap="plasma",
        s=POINT_SIZE,
        label="frame",
    )
    if len(frames) > 0:
        ax.text(*trajectory_3d[0], f"{int(frames[0])}", color=TRAJECTORY_COLOR)
        ax.text(*trajectory_3d[-1], f"{int(frames[-1])}", color=TRAJECTORY_COLOR)

    camera_points = draw_camera_matplotlib(ax, K, image_size, all_points)
    all_points = np.vstack([all_points, camera_points])
    ax.set_xlabel("X camera")
    ax.set_ylabel("Y camera")
    ax.set_zlabel("Z camera")
    ax.set_title(f"Jumper trajectory on G_30, ref frame {REFERENCE_FRAME}")
    ax.legend(loc="upper right")
    set_axes_equal(ax, all_points)
    ax.view_init(elev=22, azim=-62)
    try:
        ax.figure.colorbar(scatter, ax=ax, shrink=0.65, pad=0.08, label="ordine temporale")
    except Exception:
        pass


# Display the trajectory plot
def show_matplotlib_viewer(
    slope: dict,
    trajectory_3d: np.ndarray,
    frames: np.ndarray,
    K: np.ndarray,
    image_size: tuple[int, int],
) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    draw_scene_matplotlib(ax, slope, trajectory_3d, frames, K, image_size)
    fig.tight_layout()
    print("Viewer Matplotlib aperto. Usa mouse e toolbar per ruotare, zoomare e spostarti.")
    plt.show()


# Build lines for three dimensional display
def open3d_line_set(o3d, points: np.ndarray, lines: list[list[int]], colors: list[tuple[float, float, float]]):
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    line_set.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=float))
    return line_set


# Build camera display geometry
def camera_open3d_geometry(o3d, K: np.ndarray, image_size: tuple[int, int], scene_points: np.ndarray):
    corners = camera_frustum_points(K, image_size, scene_points)
    axis_points = camera_axis_points(scene_points)
    points = np.vstack([np.zeros((1, 3), dtype=float), corners, axis_points[1:]])
    lines = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 1],
        [0, 5],
        [0, 6],
        [0, 7],
    ]
    colors = [CAMERA_COLOR] * 8 + [CAMERA_X_COLOR, CAMERA_Y_COLOR, CAMERA_Z_COLOR]
    return open3d_line_set(o3d, points, lines, colors)


# Display the interactive scene
def show_open3d_viewer(
    slope: dict,
    trajectory_3d: np.ndarray,
    K: np.ndarray,
    image_size: tuple[int, int],
) -> None:
    try:
        import open3d as o3d
    except ImportError as error:
        raise RuntimeError("Open3D non installato: usa venv/bin/python -m pip install open3d") from error

    left = slope["left_3d"]
    right = slope["right_3d"]
    center = slope["center_3d"]
    points_blocks = [left, right, center, trajectory_3d]
    count_left = len(left)
    count_center = len(center)
    count_traj = len(trajectory_3d)
    offset_right = count_left
    offset_center = count_left + len(right)
    offset_traj = offset_center + count_center

    lines = []
    colors = []
    lines.extend([[idx, idx + 1] for idx in range(count_left - 1)])
    colors.extend([LEFT_COLOR] * (count_left - 1))
    lines.extend([[offset_right + idx, offset_right + idx + 1] for idx in range(len(right) - 1)])
    colors.extend([RIGHT_COLOR] * (len(right) - 1))
    lines.extend([[offset_center + idx, offset_center + idx + 1] for idx in range(count_center - 1)])
    colors.extend([CENTER_COLOR] * (count_center - 1))
    lines.extend([[offset_traj + idx, offset_traj + idx + 1] for idx in range(count_traj - 1)])
    colors.extend([TRAJECTORY_COLOR] * max(0, count_traj - 1))

    stride = max(1, count_left // 25)
    for idx in range(0, count_left, stride):
        lines.append([idx, offset_right + idx])
        colors.append(RULING_COLOR)

    scene_points = np.vstack(points_blocks)
    geometries = [open3d_line_set(o3d, scene_points, lines, colors)]
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(trajectory_3d)
    point_colors = np.tile(np.asarray(TRAJECTORY_COLOR, dtype=float), (len(trajectory_3d), 1))
    point_cloud.colors = o3d.utility.Vector3dVector(point_colors)
    geometries.append(point_cloud)
    geometries.append(camera_open3d_geometry(o3d, K, image_size, scene_points))

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(window_name="Jumper trajectory 3D", width=1200, height=850)
    for geometry in geometries:
        visualizer.add_geometry(geometry)

    options = visualizer.get_render_option()
    options.background_color = np.asarray(BACKGROUND_COLOR, dtype=float)
    options.point_size = POINT_SIZE
    options.line_width = 2.0
    print("Viewer Open3D aperto. Mouse sinistro ruota, rotella zoom, mouse destro o shift+sinistro sposta.")
    visualizer.run()
    visualizer.destroy_window()


# Select the requested viewer
def maybe_show_viewer(
    slope: dict,
    trajectory_3d: np.ndarray,
    frames: np.ndarray,
    K: np.ndarray,
    image_size: tuple[int, int],
) -> None:
    if not OPEN_VIEWER:
        return

    backend = VIEWER_BACKEND.lower().strip()
    if backend not in {"auto", "open3d", "matplotlib"}:
        raise ValueError("VIEWER_BACKEND deve essere 'auto', 'open3d' o 'matplotlib'.")

    if backend in {"auto", "open3d"}:
        try:
            show_open3d_viewer(slope, trajectory_3d, K, image_size)
            return
        except RuntimeError as error:
            if backend == "open3d":
                raise
            print(f"{error}. Uso Matplotlib come fallback.")

    show_matplotlib_viewer(slope, trajectory_3d, frames, K, image_size)


# Run trajectory reconstruction
def run_pipeline() -> dict:
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video non trovato: {VIDEO_PATH}")
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV non trovato: {CSV_PATH}")

    changes, intervals, fps, total_frames = detect_camera_intervals(VIDEO_PATH)
    save_camera_intervals(CAMERA_INTERVALS_JSON_PATH, VIDEO_PATH, changes, intervals, fps, total_frames)
    interval = selected_interval(intervals, SELECTED_CAMERA_ID, REFERENCE_FRAME)
    start_frame = interval.start.frame_idx
    end_frame = interval.end.frame_idx

    print(f"Video: {VIDEO_PATH}", flush=True)
    print(f"FPS: {fps:.3f}, frame totali: {total_frames}", flush=True)
    print(switchCam.format_camera_intervals(intervals), flush=True)
    print(f"Uso camera {SELECTED_CAMERA_ID}: frame {start_frame}-{end_frame}", flush=True)
    print(f"Frame riferimento: {REFERENCE_FRAME}", flush=True)

    provider = JumperProvider(str(CSV_PATH))
    kalman_points = read_kalman_points(CSV_PATH, start_frame, end_frame)
    pairwise_warps, motion_rows = estimate_pairwise_homographies(VIDEO_PATH, provider, start_frame, end_frame, WARP_MODE)
    transforms = compute_frame_to_reference_transforms(pairwise_warps, start_frame, end_frame, REFERENCE_FRAME)
    cumulative_scores = compute_cumulative_motion_scores(motion_rows, start_frame, end_frame, REFERENCE_FRAME)
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
    image_size = image_size_from_video(VIDEO_PATH)
    plot_mask = motion_scores >= MIN_RELIABLE_CUMULATIVE_SCORE if PLOT_ONLY_RELIABLE_POINTS else np.ones(len(frames), dtype=bool)
    if not np.any(plot_mask):
        plot_mask = np.ones(len(frames), dtype=bool)
    plot_frames = frames[plot_mask]
    plot_trajectory_3d = trajectory_3d[plot_mask]

    save_reference_points_csv(
        REFERENCE_POINTS_CSV_PATH,
        frames,
        image_points,
        reference_points,
        plane_coords,
        trajectory_3d,
        motion_scores,
        fps,
    )
    save_trajectory_npz(
        TRAJECTORY_NPZ_PATH,
        frames,
        image_points,
        reference_points,
        plane_coords,
        trajectory_3d,
        motion_scores,
        center_plane,
    )
    if SAVE_STATIC_PLOT:
        plot_static_3d(TRAJECTORY_PNG_PATH, slope, plot_trajectory_3d, plot_frames, K, image_size)

    print(f"Punti traiettoria validi: {len(frames)}", flush=True)
    print(f"Punti affidabili score>={MIN_RELIABLE_CUMULATIVE_SCORE}: {int(np.sum(motion_scores >= MIN_RELIABLE_CUMULATIVE_SCORE))}", flush=True)
    print(f"Frame iniziale/finale traiettoria: {int(frames[0])}-{int(frames[-1])}", flush=True)
    print(f"Output intervalli: {CAMERA_INTERVALS_JSON_PATH}", flush=True)
    print(f"Output punti: {REFERENCE_POINTS_CSV_PATH}", flush=True)
    print(f"Output 3D: {TRAJECTORY_NPZ_PATH}", flush=True)
    if SAVE_STATIC_PLOT:
        print(f"Plot 3D: {TRAJECTORY_PNG_PATH}", flush=True)

    maybe_show_viewer(slope, plot_trajectory_3d, plot_frames, K, image_size)
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


# Run the trajectory workflow
def main() -> int:
    run_pipeline()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
