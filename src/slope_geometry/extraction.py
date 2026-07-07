# Slope reconstruction workflow
# Step one loads the calibration matrix from JSON
# Step two lets the user annotate normal direction segments to the slope's plane and the two projected track curves
# Step three estimates the normal vanishing point from the annotated normal segments
# Step four converts that vanishing point into a camera space normal direction
# Step five samples both curves and matches corresponding points through lines toward the vanishing point
# Step six reconstructs two parallel three dimensional curves separated by TRACK_WIDTH
# Step seven saves annotations overlays correspondences CSV NPZ data and a static plot
# Option TRACK_WIDTH sets the separation between the two reconstructed parallel planes
# Option NUM_SAMPLES controls the number of reconstructed correspondence points
# Option DISPLAY_SCALE controls the annotation window scale
# Option MIN_NORMAL_SEGMENTS controls the minimum normal segment count
# Option DENSE_CURVE_SAMPLES controls the dense matching curve resolution
# Option ARC_LENGTH_REGULARIZATION_PX keeps matching ordered along the curve
# Formula image line from two points is [y1-y2 x2-x1 x1*y2-x2*y1]
# Formula vanishing point is the last singular vector of the normal line matrix
# Formula normal direction is normalize(inv(K) @ vanishing_point)
# Formula point to line distance is abs(a x + b y + c) / sqrt(a^2 + b^2)
# Formula matching cost is line distance plus regularization times absolute arc parameter difference
# Formula plane depths are depth_a = offset / dot(ray_a n) and depth_b = (offset + width) / dot(ray_b n)
# Formula reconstruction residual is (right - left) - width n

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

DEFAULT_FRAME_PATH = Path("outputGeometry/selected_frame.jpg")

try:
    from calibration_phase import FRAME_PATH as CALIBRATION_FRAME_PATH
except Exception:
    CALIBRATION_FRAME_PATH = DEFAULT_FRAME_PATH

from src.track_geometry.curve_sampling import resample_polyline
from src.track_geometry.models import CameraCalibration
from src.track_geometry.reconstruction import (
    estimate_vanishing_point_from_segments as estimate_vanishing_point_from_segments_core,
    reconstruct_corresponding_rays,
)


FRAME_PATH = Path(CALIBRATION_FRAME_PATH)
K_JSON_PATH = Path("outputGeometry/calibration_K.json")

TRACK_WIDTH = 1.0
NUM_SAMPLES = 80
DISPLAY_SCALE = 0.75
MIN_NORMAL_SEGMENTS = 2

ANNOTATIONS_JSON_PATH = Path("outputGeometry/slope_extraction_annotations.json")
OVERLAY_PATH = Path("outputGeometry/slope_extraction_overlay.jpg")
CORRESPONDENCES_PATH = Path("outputGeometry/slope_extraction_correspondences.jpg")
RECONSTRUCTION_CSV_PATH = Path("outputGeometry/slope_extraction_reconstruction.csv")
RECONSTRUCTION_NPZ_PATH = Path("outputGeometry/slope_extraction_reconstruction.npz")
RECONSTRUCTION_3D_PATH = Path("outputGeometry/slope_extraction_3d.png")

DENSE_CURVE_SAMPLES = 1600
ARC_LENGTH_REGULARIZATION_PX = 8.0
WINDOW_NAME = "slope_extraction"


@dataclass
class AnnotationState:
    phase: str = "normals"
    normal_segments: list[tuple[tuple[float, float], tuple[float, float]]] = field(default_factory=list)
    pending_normal_point: tuple[float, float] | None = None
    curve_a: list[tuple[float, float]] = field(default_factory=list)
    curve_b: list[tuple[float, float]] = field(default_factory=list)
    done: bool = False


@dataclass
class CorrespondenceResult:
    points_a: np.ndarray
    points_b: np.ndarray
    errors: np.ndarray
    mean_error: float
    reversed_curve_b: bool


@dataclass
class ReconstructionResult:
    left_3d: np.ndarray
    right_3d: np.ndarray
    center_3d: np.ndarray
    residuals: np.ndarray
    depths: np.ndarray
    normal: np.ndarray
    positive_depth_ratio: float
    mean_residual: float


# Load the camera matrix
def load_camera_matrix(path: Path = K_JSON_PATH) -> np.ndarray:
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if "K" in data:
            return np.asarray(data["K"], dtype=float)
    raise FileNotFoundError(f"Camera matrix JSON non trovato: {path}")
    return np.eye(3, dtype=float)


# Normalize a vector
def unit(vector: np.ndarray, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError(f"{name} ha norma quasi nulla.")
    return vector / norm


# Build a line from points
def line_from_points(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    x1, y1 = [float(value) for value in first]
    x2, y2 = [float(value) for value in second]
    line = np.array([y1 - y2, x2 - x1, x1 * y2 - x2 * y1], dtype=float)
    return unit(line, "linea immagine")


# Build a line from a segment
def line_from_segment(segment: tuple[tuple[float, float], tuple[float, float]]) -> np.ndarray:
    return line_from_points(np.asarray(segment[0], dtype=float), np.asarray(segment[1], dtype=float))


# Estimate a vanishing point
def estimate_vanishing_point_from_segments(
    segments: list[tuple[tuple[float, float], tuple[float, float]]],
) -> np.ndarray:
    return estimate_vanishing_point_from_segments_core(segments, min_segments=MIN_NORMAL_SEGMENTS)


# Recover a camera direction
def camera_direction_from_vanishing_point(K: np.ndarray, vanishing_point: np.ndarray) -> np.ndarray:
    direction = np.linalg.inv(K) @ np.asarray(vanishing_point, dtype=float).reshape(3)
    return unit(direction, "direzione normale")


# Compute plane viewing angles
def plane_camera_angles_degrees(normal: np.ndarray) -> tuple[float, float]:
    optical_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    normal = unit(normal, "normale")
    dot_value = abs(float(np.dot(normal, optical_axis)))
    dot_value = min(1.0, max(-1.0, dot_value))
    normal_axis = math.degrees(math.acos(dot_value))
    plane_axis = 90.0 - normal_axis
    return normal_axis, plane_axis


# Compute distances from a line
def homogeneous_line_distance(line: np.ndarray, points: np.ndarray) -> np.ndarray:
    denom = float(np.linalg.norm(line[:2]))
    if denom < 1e-12:
        raise ValueError("Linea omogenea degenerata.")
    return np.abs(points @ line[:2] + line[2]) / denom


# Build a ruling image line
def line_through_point_and_vanishing(point: np.ndarray, vanishing_point: np.ndarray) -> np.ndarray:
    point_h = np.array([point[0], point[1], 1.0], dtype=float)
    line = np.cross(point_h, vanishing_point)
    return unit(line, "linea di corrispondenza")


# Densely sample curve points
def dense_resample(points: np.ndarray, sample_count: int = DENSE_CURVE_SAMPLES) -> np.ndarray:
    count = max(sample_count, len(points) * 40, NUM_SAMPLES * 10)
    return resample_polyline(np.asarray(points, dtype=float), count)


# Match a point on another curve
def monotonic_match_curve(
    points_a: np.ndarray,
    dense_b: np.ndarray,
    vanishing_point: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(points_a)
    m = len(dense_b)
    if m < n:
        dense_b = resample_polyline(dense_b, n)
        m = len(dense_b)

    costs = np.zeros((n, m), dtype=float)
    raw_distances = np.zeros((n, m), dtype=float)
    normalized_arc = np.linspace(0.0, 1.0, m)

    for idx, point in enumerate(points_a):
        line = line_through_point_and_vanishing(point, vanishing_point)
        distances = homogeneous_line_distance(line, dense_b)
        target = idx / max(1, n - 1)
        costs[idx] = distances + ARC_LENGTH_REGULARIZATION_PX * np.abs(normalized_arc - target)
        raw_distances[idx] = distances

    dp = np.full((n, m), np.inf, dtype=float)
    back = np.full((n, m), -1, dtype=int)
    dp[0] = costs[0]

    for i in range(1, n):
        best_value = np.inf
        best_index = -1
        for j in range(m):
            if j > 0 and dp[i - 1, j - 1] < best_value:
                best_value = dp[i - 1, j - 1]
                best_index = j - 1
            if best_index >= 0:
                dp[i, j] = costs[i, j] + best_value
                back[i, j] = best_index

    end = int(np.argmin(dp[-1]))
    if not np.isfinite(dp[-1, end]):
        indices = np.linspace(0, m - 1, n).round().astype(int)
        return dense_b[indices], raw_distances[np.arange(n), indices]

    indices = np.empty(n, dtype=int)
    indices[-1] = end
    for i in range(n - 1, 0, -1):
        indices[i - 1] = back[i, indices[i]]

    return dense_b[indices], raw_distances[np.arange(n), indices]


# Build curve point correspondences
def build_vanishing_correspondences(
    curve_a: np.ndarray,
    curve_b: np.ndarray,
    vanishing_point: np.ndarray,
    num_samples: int = NUM_SAMPLES,
) -> CorrespondenceResult:
    points_a = resample_polyline(np.asarray(curve_a, dtype=float), num_samples)
    candidates = []

    for reversed_curve in (False, True):
        source_b = np.asarray(curve_b, dtype=float)
        if reversed_curve:
            source_b = source_b[::-1]
        dense_b = dense_resample(source_b)
        points_b, errors = monotonic_match_curve(points_a, dense_b, vanishing_point)
        candidates.append(
            CorrespondenceResult(
                points_a=points_a,
                points_b=points_b,
                errors=errors,
                mean_error=float(np.mean(errors)),
                reversed_curve_b=reversed_curve,
            ),
        )

    candidates.sort(key=lambda item: item.mean_error)
    return candidates[0]


# Reconstruct curves in space
def reconstruct_curves_3d(
    camera: CameraCalibration,
    points_a: np.ndarray,
    points_b: np.ndarray,
    normal: np.ndarray,
    width: float = TRACK_WIDTH,
) -> ReconstructionResult:
    left_3d, right_3d, residuals, depths, oriented_normal = reconstruct_corresponding_rays(
        camera,
        points_a,
        points_b,
        normal,
        width,
    )
    return ReconstructionResult(
        left_3d=left_3d,
        right_3d=right_3d,
        center_3d=0.5 * (left_3d + right_3d),
        residuals=residuals,
        depths=depths,
        normal=oriented_normal,
        positive_depth_ratio=float(np.mean(depths > 0.0)),
        mean_residual=float(np.mean(np.linalg.norm(residuals, axis=1))),
    )


# Draw an image polyline
def draw_polyline(image: np.ndarray, points: list[tuple[float, float]], color: tuple[int, int, int]) -> None:
    if not points:
        return
    pts = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
    if len(points) >= 2:
        cv2.polylines(image, [pts], False, color, 2, cv2.LINE_AA)
    for idx, point in enumerate(points):
        cv2.circle(image, rounded_point(point), 4, color, -1, cv2.LINE_AA)
        cv2.putText(
            image,
            str(idx),
            (int(point[0]) + 5, int(point[1]) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )


# Round a point to pixels
def rounded_point(point: tuple[float, float] | np.ndarray) -> tuple[int, int]:
    return int(round(float(point[0]))), int(round(float(point[1])))


# Render current annotations
def render_annotation(frame: np.ndarray, state: AnnotationState) -> np.ndarray:
    overlay = frame.copy()
    for first, second in state.normal_segments:
        cv2.line(overlay, rounded_point(first), rounded_point(second), (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(overlay, rounded_point(first), 4, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(overlay, rounded_point(second), 4, (0, 255, 255), -1, cv2.LINE_AA)

    if state.pending_normal_point is not None:
        cv2.circle(overlay, rounded_point(state.pending_normal_point), 5, (0, 200, 255), -1, cv2.LINE_AA)

    draw_polyline(overlay, state.curve_a, (0, 255, 0))
    draw_polyline(overlay, state.curve_b, (0, 0, 255))

    if state.phase == "normals":
        text = "Normali: click coppie di punti. Enter conferma, u undo, r reset, q esce."
    elif state.phase == "curve_a":
        text = "Polyline A: click punti ordinati. Enter conferma, u undo, r reset, q esce."
    else:
        text = "Polyline B: click punti ordinati. Enter ricostruisce, u undo, r reset, q esce."
    cv2.putText(overlay, text, (25, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return overlay


# Scale an image for display
def scaled_for_display(image: np.ndarray) -> np.ndarray:
    if abs(DISPLAY_SCALE - 1.0) < 1e-9:
        return image
    return cv2.resize(image, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE, interpolation=cv2.INTER_AREA)


# Convert display coordinates to image coordinates
def display_to_image_point(x: int, y: int, frame_shape: tuple[int, ...]) -> tuple[float, float]:
    px = float(x) / DISPLAY_SCALE
    py = float(y) / DISPLAY_SCALE
    px = min(max(px, 0.0), float(frame_shape[1] - 1))
    py = min(max(py, 0.0), float(frame_shape[0] - 1))
    return px, py


# Reset the active annotation phase
def reset_current_phase(state: AnnotationState) -> None:
    if state.phase == "normals":
        state.normal_segments.clear()
        state.pending_normal_point = None
    elif state.phase == "curve_a":
        state.curve_a.clear()
    else:
        state.curve_b.clear()


# Undo the latest annotation
def undo_current_phase(state: AnnotationState) -> None:
    if state.phase == "normals":
        if state.pending_normal_point is not None:
            state.pending_normal_point = None
        elif state.normal_segments:
            state.normal_segments.pop()
    elif state.phase == "curve_a" and state.curve_a:
        state.curve_a.pop()
    elif state.phase == "curve_b" and state.curve_b:
        state.curve_b.pop()


# Advance the annotation phase
def advance_phase(state: AnnotationState) -> None:
    if state.phase == "normals":
        if len(state.normal_segments) < MIN_NORMAL_SEGMENTS:
            print(f"Servono almeno {MIN_NORMAL_SEGMENTS} segmenti normali.")
            return
        state.pending_normal_point = None
        state.phase = "curve_a"
    elif state.phase == "curve_a":
        if len(state.curve_a) < 2:
            print("La polyline A deve avere almeno 2 punti.")
            return
        state.phase = "curve_b"
    else:
        if len(state.curve_b) < 2:
            print("La polyline B deve avere almeno 2 punti.")
            return
        state.done = True


# Collect annotations interactively
def run_interactive_annotation(frame: np.ndarray) -> AnnotationState:
    state = AnnotationState()

    # Handle annotation mouse input
    def on_mouse(event, x, y, flags, userdata) -> None:
        del flags, userdata
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        point = display_to_image_point(x, y, frame.shape)
        if state.phase == "normals":
            if state.pending_normal_point is None:
                state.pending_normal_point = point
            else:
                state.normal_segments.append((state.pending_normal_point, point))
                state.pending_normal_point = None
        elif state.phase == "curve_a":
            state.curve_a.append(point)
        else:
            state.curve_b.append(point)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    while not state.done:
        cv2.imshow(WINDOW_NAME, scaled_for_display(render_annotation(frame, state)))
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10):
            advance_phase(state)
        elif key == ord("u"):
            undo_current_phase(state)
        elif key == ord("r"):
            reset_current_phase(state)
        elif key in (ord("q"), 27):
            cv2.destroyWindow(WINDOW_NAME)
            raise RuntimeError("Annotazione annullata.")

    cv2.destroyWindow(WINDOW_NAME)
    return state


# Draw the completed annotation overlay
def draw_final_overlay(
    frame: np.ndarray,
    state: AnnotationState,
    correspondences: CorrespondenceResult,
    vanishing_point: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    overlay = render_annotation(frame, state)
    cv2.putText(overlay, "Annotazioni finali", (25, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    corr = frame.copy()
    draw_polyline(corr, [tuple(point) for point in state.curve_a], (0, 255, 0))
    draw_polyline(corr, [tuple(point) for point in state.curve_b], (0, 0, 255))

    stride = max(1, len(correspondences.points_a) // 25)
    for idx in range(0, len(correspondences.points_a), stride):
        a = rounded_point(correspondences.points_a[idx])
        b = rounded_point(correspondences.points_b[idx])
        cv2.line(corr, a, b, (160, 160, 160), 1, cv2.LINE_AA)
        cv2.circle(corr, a, 3, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(corr, b, 3, (0, 0, 255), -1, cv2.LINE_AA)

    if abs(float(vanishing_point[2])) > 1e-12:
        vp = vanishing_point[:2] / vanishing_point[2]
        if -frame.shape[1] <= vp[0] <= 2 * frame.shape[1] and -frame.shape[0] <= vp[1] <= 2 * frame.shape[0]:
            cv2.drawMarker(corr, rounded_point(vp), (255, 0, 255), cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)

    return overlay, corr


# Save annotation data
def save_annotations(
    path: Path,
    state: AnnotationState,
    K: np.ndarray,
    vanishing_point: np.ndarray,
    normal_direction: np.ndarray,
    normal_axis_angle: float,
    plane_axis_angle: float,
    correspondences: CorrespondenceResult,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "frame_path": str(FRAME_PATH),
        "K": K.tolist(),
        "track_width": TRACK_WIDTH,
        "normal_segments": [
            [[float(first[0]), float(first[1])], [float(second[0]), float(second[1])]]
            for first, second in state.normal_segments
        ],
        "curve_a": [[float(x), float(y)] for x, y in state.curve_a],
        "curve_b": [[float(x), float(y)] for x, y in state.curve_b],
        "normal_vanishing_point": vanishing_point.tolist(),
        "normal_direction_camera": normal_direction.tolist(),
        "normal_axis_angle_degrees": normal_axis_angle,
        "plane_axis_angle_degrees": plane_axis_angle,
        "correspondence_mean_error_px": correspondences.mean_error,
        "curve_b_was_reversed": correspondences.reversed_curve_b,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


# Save reconstruction rows
def save_reconstruction_csv(path: Path, correspondences: CorrespondenceResult, reconstruction: ReconstructionResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    residual_norms = np.linalg.norm(reconstruction.residuals, axis=1)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            "idx,a_u,a_v,b_u,b_v,left_x,left_y,left_z,right_x,right_y,right_z,"
            "center_x,center_y,center_z,residual_norm\n",
        )
        for idx, (a, b, left, right, center, residual) in enumerate(
            zip(
                correspondences.points_a,
                correspondences.points_b,
                reconstruction.left_3d,
                reconstruction.right_3d,
                reconstruction.center_3d,
                residual_norms,
            ),
        ):
            values = [idx, *a, *b, *left, *right, *center, residual]
            handle.write(",".join(str(float(value)) if idx2 > 0 else str(value) for idx2, value in enumerate(values)))
            handle.write("\n")


# Save reconstruction arrays
def save_reconstruction_npz(path: Path, correspondences: CorrespondenceResult, reconstruction: ReconstructionResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        points_a=correspondences.points_a,
        points_b=correspondences.points_b,
        errors=correspondences.errors,
        left_3d=reconstruction.left_3d,
        right_3d=reconstruction.right_3d,
        center_3d=reconstruction.center_3d,
        residuals=reconstruction.residuals,
        depths=reconstruction.depths,
        normal=reconstruction.normal,
        track_width=np.asarray([TRACK_WIDTH], dtype=float),
    )


# Equalize plot axes
def set_axes_equal(ax, points: np.ndarray) -> None:
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    radius = max(radius, 1e-6)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


# Plot reconstructed slope curves
def plot_reconstruction_3d(path: Path, reconstruction: ReconstructionResult) -> None:
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError as error:
        raise RuntimeError("matplotlib non installato: usa venv/bin/python -m pip install matplotlib") from error

    path.parent.mkdir(parents=True, exist_ok=True)
    fig = Figure(figsize=(10, 7))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection="3d")
    left = reconstruction.left_3d
    right = reconstruction.right_3d
    center = reconstruction.center_3d

    ax.plot(left[:, 0], left[:, 1], left[:, 2], label="polyline A")
    ax.plot(right[:, 0], right[:, 1], right[:, 2], label="polyline B")
    ax.plot(center[:, 0], center[:, 1], center[:, 2], "--", label="center")

    stride = max(1, len(left) // 25)
    for idx in range(0, len(left), stride):
        segment = np.vstack([left[idx], right[idx]])
        ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color="0.55", linewidth=0.8)

    all_points = np.vstack([left, right])
    set_axes_equal(ax, all_points)
    ax.set_xlabel("X camera")
    ax.set_ylabel("Y camera")
    ax.set_zlabel("Z camera")
    ax.set_title("Slope parallel curves reconstruction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)


# Run slope reconstruction
def run_pipeline(frame: np.ndarray, state: AnnotationState, K: np.ndarray) -> tuple[CorrespondenceResult, ReconstructionResult]:
    normal_vp = estimate_vanishing_point_from_segments(state.normal_segments)
    normal_direction = camera_direction_from_vanishing_point(K, normal_vp)
    normal_axis_angle, plane_axis_angle = plane_camera_angles_degrees(normal_direction)
    correspondences = build_vanishing_correspondences(
        np.asarray(state.curve_a, dtype=float),
        np.asarray(state.curve_b, dtype=float),
        normal_vp,
        num_samples=NUM_SAMPLES,
    )
    reconstruction = reconstruct_curves_3d(
        CameraCalibration(K),
        correspondences.points_a,
        correspondences.points_b,
        normal_direction,
        width=TRACK_WIDTH,
    )

    overlay, corr_overlay = draw_final_overlay(frame, state, correspondences, normal_vp)
    OVERLAY_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OVERLAY_PATH), overlay)
    cv2.imwrite(str(CORRESPONDENCES_PATH), corr_overlay)
    save_annotations(
        ANNOTATIONS_JSON_PATH,
        state,
        K,
        normal_vp,
        reconstruction.normal,
        normal_axis_angle,
        plane_axis_angle,
        correspondences,
    )
    save_reconstruction_csv(RECONSTRUCTION_CSV_PATH, correspondences, reconstruction)
    save_reconstruction_npz(RECONSTRUCTION_NPZ_PATH, correspondences, reconstruction)
    plot_reconstruction_3d(RECONSTRUCTION_3D_PATH, reconstruction)

    print("Punto di fuga normali:", normal_vp)
    print("Direzione normale camera:", reconstruction.normal)
    print(f"Angolo normale-asse ottico: {normal_axis_angle:.3f} deg")
    print(f"Angolo piano-asse ottico: {plane_axis_angle:.3f} deg")
    print(f"Errore medio corrispondenze: {correspondences.mean_error:.3f} px")
    print(f"Residual medio 3D: {reconstruction.mean_residual:.6f}")
    print(f"Positive depth ratio: {reconstruction.positive_depth_ratio:.3f}")
    return correspondences, reconstruction


# Run the slope workflow
def main() -> int:
    frame = cv2.imread(str(FRAME_PATH))
    if frame is None:
        raise FileNotFoundError(f"Frame non trovato: {FRAME_PATH}")

    K = load_camera_matrix(K_JSON_PATH)
    print("K usata:")
    print(K)
    print("Fasi: segmenti normali, polyline A, polyline B.")
    print("Click sinistro aggiunge punti; Enter conferma; u undo; r reset; q esce.")

    state = run_interactive_annotation(frame)
    run_pipeline(frame, state, K)

    print(f"Annotazioni salvate: {ANNOTATIONS_JSON_PATH}")
    print(f"Overlay salvato: {OVERLAY_PATH}")
    print(f"Corrispondenze salvate: {CORRESPONDENCES_PATH}")
    print(f"CSV 3D salvato: {RECONSTRUCTION_CSV_PATH}")
    print(f"NPZ 3D salvato: {RECONSTRUCTION_NPZ_PATH}")
    print(f"Plot 3D salvato: {RECONSTRUCTION_3D_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
