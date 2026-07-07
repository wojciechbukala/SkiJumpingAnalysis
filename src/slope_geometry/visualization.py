# Slope visualization workflow
# Step one loads slope reconstruction arrays and annotation metadata
# Step two optionally recomputes corrected parallel plane geometry from image correspondences and K
# Step three builds plane bases and plane to image homographies for both track borders
# Step four saves border and center-plane homographies verification metrics and per point plane coordinates
# Step five displays the reconstruction with Matplotlib or Open3D
# Option VIEWER_BACKEND chooses auto open3d or matplotlib
# Option USE_HOMOGRAPHY_CORRECTED_POINTS selects corrected lifted points for display
# Options SHOW_CENTERLINE SHOW_RULINGS SHOW_NORMAL SHOW_POINTS SHOW_AXES SHOW_CAMERA and SHOW_PLANE_BASES toggle scene elements
# Options RULING_STRIDE NORMAL_SCALE CAMERA_FRUSTUM_SCALE CAMERA_AXIS_SCALE and PLANE_BASIS_SCALE tune visualization geometry
# Formula camera ray is normalize(inv(K) @ [u v 1])
# Formula projection is image = K @ P followed by division by z
# Formula plane basis uses e1 from projected curve direction and e2 = cross(normal e1)
# Formula plane homography is H = K @ [e1 e2 origin]
# Formula plane point is P = origin + u e1 + v e2
# Formula translation residual is (right - left) - width normal

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


RECONSTRUCTION_NPZ_PATH = Path("outputGeometry/slope_extraction_reconstruction.npz")
ANNOTATIONS_JSON_PATH = Path("outputGeometry/slope_extraction_annotations.json")
HOMOGRAPHY_JSON_PATH = Path("outputGeometry/slope_plane_homographies.json")
HOMOGRAPHY_POINTS_CSV_PATH = Path("outputGeometry/slope_plane_points.csv")

VIEWER_BACKEND = "auto"
WINDOW_TITLE = "Slope 3D reconstruction"

USE_HOMOGRAPHY_CORRECTED_POINTS = True
SHOW_CENTERLINE = True
SHOW_RULINGS = True
SHOW_NORMAL = True
SHOW_POINTS = True
SHOW_AXES = True
SHOW_CAMERA = True
SHOW_PLANE_BASES = True

RULING_STRIDE = 4
NORMAL_SCALE = 2.0
CAMERA_FRUSTUM_SCALE = 0.28
CAMERA_AXIS_SCALE = 0.45
PLANE_BASIS_SCALE = 0.9
POINT_SIZE = 5.0
MATPLOTLIB_LINE_WIDTH = 2.0

BACKGROUND_COLOR = (0.04, 0.045, 0.055)
LEFT_COLOR = (0.1, 0.85, 0.25)
RIGHT_COLOR = (0.95, 0.18, 0.16)
CENTER_COLOR = (0.2, 0.55, 1.0)
RULING_COLOR = (0.68, 0.68, 0.68)
NORMAL_COLOR = (1.0, 0.75, 0.1)
CAMERA_COLOR = (0.0, 0.0, 0.0)
CAMERA_X_COLOR = (1.0, 0.18, 0.18)
CAMERA_Y_COLOR = (0.18, 0.9, 0.18)
CAMERA_Z_COLOR = (0.25, 0.48, 1.0)
PLANE_U_COLOR = (0.0, 0.95, 0.95)
PLANE_V_COLOR = (1.0, 0.25, 0.95)


# Load reconstructed slope data
def load_reconstruction(path: Path = RECONSTRUCTION_NPZ_PATH) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Ricostruzione non trovata: {path}. Prima esegui slope_extraction.py")

    with np.load(path) as data:
        required = ["left_3d", "right_3d", "center_3d", "normal", "track_width"]
        missing = [name for name in required if name not in data.files]
        if missing:
            raise ValueError(f"File NPZ incompleto, mancano: {missing}")

        reconstruction = {name: np.asarray(data[name], dtype=float) for name in data.files}

    left = reconstruction["left_3d"]
    right = reconstruction["right_3d"]
    center = reconstruction["center_3d"]
    if left.ndim != 2 or right.ndim != 2 or center.ndim != 2:
        raise ValueError("left_3d, right_3d e center_3d devono essere array 2D.")
    if left.shape[1] != 3 or right.shape[1] != 3 or center.shape[1] != 3:
        raise ValueError("Le curve 3D devono avere forma (N, 3).")
    if len(left) != len(right) or len(left) != len(center):
        raise ValueError("left_3d, right_3d e center_3d devono avere la stessa lunghezza.")
    if len(left) < 2:
        raise ValueError("Servono almeno due punti 3D per visualizzare una curva.")

    return reconstruction


# Load annotation data
def load_annotations(path: Path = ANNOTATIONS_JSON_PATH) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# Normalize a vector
def unit(vector: np.ndarray, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError(f"{name} ha norma quasi nulla.")
    return vector / norm


# Read the camera matrix
def camera_matrix_from_annotations(annotations: dict) -> np.ndarray:
    if "K" not in annotations:
        raise ValueError("K non trovata nelle annotazioni: serve per camera e omografie.")
    K = np.asarray(annotations["K"], dtype=float)
    if K.shape != (3, 3):
        raise ValueError("K nelle annotazioni deve avere forma 3x3.")
    return K


# Read the image dimensions
def image_size_from_annotations(annotations: dict, K: np.ndarray) -> tuple[int, int]:
    frame_path = annotations.get("frame_path")
    if frame_path is not None and Path(frame_path).exists():
        try:
            import cv2

            frame = cv2.imread(str(frame_path))
            if frame is not None:
                height, width = frame.shape[:2]
                return int(width), int(height)
        except ImportError:
            pass

    width = max(1, int(round(2.0 * float(K[0, 2]))))
    height = max(1, int(round(2.0 * float(K[1, 2]))))
    return width, height


# Convert pixels to camera rays
def pixel_rays(K: np.ndarray, points: np.ndarray) -> np.ndarray:
    inv_K = np.linalg.inv(K)
    rays = []
    for point in np.asarray(points, dtype=float):
        ray = inv_K @ np.array([point[0], point[1], 1.0], dtype=float)
        rays.append(unit(ray, "raggio camera"))
    return np.vstack(rays)


# Project points to the image
def project_points(K: np.ndarray, points_3d: np.ndarray) -> np.ndarray:
    projected = (K @ np.asarray(points_3d, dtype=float).T).T
    return projected[:, :2] / projected[:, 2:3]


# Estimate parallel plane geometry
def estimate_parallel_plane_geometry(reconstruction: dict[str, np.ndarray], K: np.ndarray) -> dict[str, np.ndarray | float]:
    if "points_a" not in reconstruction or "points_b" not in reconstruction:
        raise ValueError("Nel file NPZ mancano points_a/points_b, necessari per stimare l'omografia.")

    points_a = reconstruction["points_a"]
    points_b = reconstruction["points_b"]
    normal = unit(reconstruction["normal"], "normale")
    width = float(np.ravel(reconstruction["track_width"])[0])
    rays_a = pixel_rays(K, points_a)
    rays_b = pixel_rays(K, points_b)
    candidates = []

    for sign in (1.0, -1.0):
        n = sign * normal
        denom_a = rays_a @ n
        denom_b = rays_b @ n
        if np.any(np.abs(denom_a) < 1e-10) or np.any(np.abs(denom_b) < 1e-10):
            continue

        lhs = []
        rhs = []
        for ray_a, ray_b, dot_a, dot_b in zip(rays_a, rays_b, denom_a, denom_b):
            lhs.extend((ray_b / dot_b - ray_a / dot_a).tolist())
            rhs.extend((width * (n - ray_b / dot_b)).tolist())

        lhs_arr = np.asarray(lhs, dtype=float).reshape(-1, 1)
        rhs_arr = np.asarray(rhs, dtype=float)
        plane_offset = float(np.linalg.lstsq(lhs_arr, rhs_arr, rcond=None)[0][0])
        depth_a = plane_offset / denom_a
        depth_b = (plane_offset + width) / denom_b
        left = rays_a * depth_a[:, None]
        right = rays_b * depth_b[:, None]
        residuals = (right - left) - width * n
        positive_depth_ratio = float(np.mean(np.concatenate([depth_a, depth_b]) > 0.0))
        mean_translation_residual = float(np.mean(np.linalg.norm(residuals, axis=1)))

        candidates.append(
            (
                positive_depth_ratio,
                -mean_translation_residual,
                {
                    "left_3d": left,
                    "right_3d": right,
                    "center_3d": 0.5 * (left + right),
                    "normal": n,
                    "track_width": np.asarray([width], dtype=float),
                    "plane_offset_a": plane_offset,
                    "plane_offset_b": plane_offset + width,
                    "translation_residuals": residuals,
                    "positive_depth_ratio": positive_depth_ratio,
                    "mean_translation_residual": mean_translation_residual,
                },
            ),
        )

    if not candidates:
        raise ValueError("Impossibile stimare i due piani: raggi quasi paralleli ai piani.")

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


# Build a plane coordinate basis
def plane_basis_from_points(points: np.ndarray, normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = unit(normal, "normale")
    direction = np.asarray(points[-1] - points[0], dtype=float)
    direction = direction - float(np.dot(direction, normal)) * normal

    if np.linalg.norm(direction) < 1e-8:
        for idx in range(1, len(points)):
            direction = np.asarray(points[idx] - points[0], dtype=float)
            direction = direction - float(np.dot(direction, normal)) * normal
            if np.linalg.norm(direction) >= 1e-8:
                break

    if np.linalg.norm(direction) < 1e-8:
        fallback = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(float(np.dot(fallback, normal))) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0], dtype=float)
        direction = fallback - float(np.dot(fallback, normal)) * normal

    e1 = unit(direction, "asse u del piano")
    e2 = unit(np.cross(normal, e1), "asse v del piano")
    return e1, e2


# Build a plane homography
def build_plane_homography(K: np.ndarray, origin: np.ndarray, e1: np.ndarray, e2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    plane_to_camera = np.column_stack([e1, e2, origin])
    H_plane_to_image = K @ plane_to_camera
    if abs(float(np.linalg.det(H_plane_to_image))) < 1e-10:
        raise ValueError("Omografia degenerata: piano troppo vicino a una configurazione singolare.")
    H_image_to_plane = np.linalg.inv(H_plane_to_image)
    return H_plane_to_image, H_image_to_plane


# Transform points with a homography
def apply_homography(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    points_h = np.column_stack([points, np.ones(len(points), dtype=float)])
    mapped = (H @ points_h.T).T
    mapped = mapped / mapped[:, 2:3]
    return mapped[:, :2]


# Convert plane coordinates to space
def plane_coordinates_to_3d(coords: np.ndarray, origin: np.ndarray, e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    return origin + coords[:, :1] * e1 + coords[:, 1:2] * e2


# Assemble plane mapping data
def build_homography_bundle(reconstruction: dict[str, np.ndarray], annotations: dict) -> dict:
    K = camera_matrix_from_annotations(annotations)
    geometry = estimate_parallel_plane_geometry(reconstruction, K)
    left = geometry["left_3d"]
    right = geometry["right_3d"]
    normal = geometry["normal"]
    width = float(np.ravel(geometry["track_width"])[0])
    e1, e2 = plane_basis_from_points(left, normal)

    origin_a = np.asarray(left[0], dtype=float)
    origin_b = origin_a + width * normal
    origin_center = 0.5 * (origin_a + origin_b)
    H_a_plane_to_image, H_a_image_to_plane = build_plane_homography(K, origin_a, e1, e2)
    H_b_plane_to_image, H_b_image_to_plane = build_plane_homography(K, origin_b, e1, e2)
    H_center_plane_to_image, H_image_to_center_plane = build_plane_homography(K, origin_center, e1, e2)

    coords_a = apply_homography(H_a_image_to_plane, reconstruction["points_a"])
    coords_b = apply_homography(H_b_image_to_plane, reconstruction["points_b"])
    lifted_a = plane_coordinates_to_3d(coords_a, origin_a, e1, e2)
    lifted_b = plane_coordinates_to_3d(coords_b, origin_b, e1, e2)
    reprojected_a = project_points(K, lifted_a)
    reprojected_b = project_points(K, lifted_b)

    reprojection_errors = np.concatenate(
        [
            np.linalg.norm(reprojected_a - reconstruction["points_a"], axis=1),
            np.linalg.norm(reprojected_b - reconstruction["points_b"], axis=1),
        ],
    )
    plane_errors = np.concatenate(
        [
            np.abs(lifted_a @ normal - float(geometry["plane_offset_a"])),
            np.abs(lifted_b @ normal - float(geometry["plane_offset_b"])),
        ],
    )
    translation_residuals = (lifted_b - lifted_a) - width * normal

    verification = {
        "mean_reprojection_error_px": float(np.mean(reprojection_errors)),
        "max_reprojection_error_px": float(np.max(reprojection_errors)),
        "mean_plane_error": float(np.mean(plane_errors)),
        "max_plane_error": float(np.max(plane_errors)),
        "normal_plane_separation": float(np.dot(origin_b - origin_a, normal)),
        "mean_translation_residual": float(np.mean(np.linalg.norm(translation_residuals, axis=1))),
        "max_translation_residual": float(np.max(np.linalg.norm(translation_residuals, axis=1))),
        "positive_depth_ratio": float(geometry["positive_depth_ratio"]),
    }

    return {
        "K": K,
        "left_3d": lifted_a,
        "right_3d": lifted_b,
        "center_3d": 0.5 * (lifted_a + lifted_b),
        "normal": normal,
        "track_width": np.asarray([width], dtype=float),
        "origin_a": origin_a,
        "origin_b": origin_b,
        "origin_center": origin_center,
        "e1": e1,
        "e2": e2,
        "H_a_plane_to_image": H_a_plane_to_image,
        "H_a_image_to_plane": H_a_image_to_plane,
        "H_b_plane_to_image": H_b_plane_to_image,
        "H_b_image_to_plane": H_b_image_to_plane,
        "H_center_plane_to_image": H_center_plane_to_image,
        "H_image_to_center_plane": H_image_to_center_plane,
        "coords_a": coords_a,
        "coords_b": coords_b,
        "verification": verification,
    }


# Save mapping output data
def save_homography_outputs(bundle: dict, reconstruction: dict[str, np.ndarray]) -> None:
    HOMOGRAPHY_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "K": bundle["K"].tolist(),
        "image_to_plane_A": bundle["H_a_image_to_plane"].tolist(),
        "plane_A_to_image": bundle["H_a_plane_to_image"].tolist(),
        "image_to_plane_B": bundle["H_b_image_to_plane"].tolist(),
        "plane_B_to_image": bundle["H_b_plane_to_image"].tolist(),
        "image_to_center_plane": bundle["H_image_to_center_plane"].tolist(),
        "center_plane_to_image": bundle["H_center_plane_to_image"].tolist(),
        "origin_A_camera": bundle["origin_a"].tolist(),
        "origin_B_camera": bundle["origin_b"].tolist(),
        "origin_center_camera": bundle["origin_center"].tolist(),
        "plane_u_axis_camera": bundle["e1"].tolist(),
        "plane_v_axis_camera": bundle["e2"].tolist(),
        "plane_normal_camera": bundle["normal"].tolist(),
        "track_width": float(np.ravel(bundle["track_width"])[0]),
        "verification": bundle["verification"],
    }
    with HOMOGRAPHY_JSON_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    with HOMOGRAPHY_POINTS_CSV_PATH.open("w", encoding="utf-8") as handle:
        handle.write(
            "idx,a_u,a_v,b_u,b_v,a_plane_u,a_plane_v,b_plane_u,b_plane_v,"
            "a_x,a_y,a_z,b_x,b_y,b_z\n",
        )
        for idx, (pixel_a, pixel_b, coord_a, coord_b, point_a, point_b) in enumerate(
            zip(
                reconstruction["points_a"],
                reconstruction["points_b"],
                bundle["coords_a"],
                bundle["coords_b"],
                bundle["left_3d"],
                bundle["right_3d"],
            ),
        ):
            values = [idx, *pixel_a, *pixel_b, *coord_a, *coord_b, *point_a, *point_b]
            handle.write(",".join(str(value) if pos == 0 else f"{float(value):.12g}" for pos, value in enumerate(values)))
            handle.write("\n")


def load_homography_bundle(path: Path = HOMOGRAPHY_JSON_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Omografie pista non trovate: {path}. Prima esegui viz_slope.py")

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    def read_matrix(json_key: str) -> np.ndarray:
        matrix = np.asarray(data[json_key], dtype=float)
        if matrix.shape != (3, 3):
            raise ValueError(f"{json_key} deve avere forma 3x3.")
        return matrix

    def read_vector(json_key: str) -> np.ndarray:
        vector = np.asarray(data[json_key], dtype=float)
        if vector.shape != (3,):
            raise ValueError(f"{json_key} deve avere forma (3,).")
        return vector

    origin_a = read_vector("origin_A_camera")
    origin_b = read_vector("origin_B_camera")
    bundle = {
        "origin_a": origin_a,
        "origin_b": origin_b,
        "origin_center": read_vector("origin_center_camera")
        if "origin_center_camera" in data
        else 0.5 * (origin_a + origin_b),
        "e1": read_vector("plane_u_axis_camera"),
        "e2": read_vector("plane_v_axis_camera"),
        "normal": read_vector("plane_normal_camera"),
        "track_width": np.asarray([float(data["track_width"])], dtype=float),
        "H_a_image_to_plane": read_matrix("image_to_plane_A"),
        "H_a_plane_to_image": read_matrix("plane_A_to_image"),
        "H_b_image_to_plane": read_matrix("image_to_plane_B"),
        "H_b_plane_to_image": read_matrix("plane_B_to_image"),
        "verification": data.get("verification", {}),
    }
    if "K" in data:
        bundle["K"] = read_matrix("K")
    if "image_to_center_plane" in data and "center_plane_to_image" in data:
        bundle["H_image_to_center_plane"] = read_matrix("image_to_center_plane")
        bundle["H_center_plane_to_image"] = read_matrix("center_plane_to_image")
    return bundle


def load_homography_corrected_reconstruction(
    reconstruction: dict[str, np.ndarray],
    path: Path = HOMOGRAPHY_POINTS_CSV_PATH,
) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Punti pista corretti non trovati: {path}. Prima esegui viz_slope.py")

    left_rows = []
    right_rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"a_x", "a_y", "a_z", "b_x", "b_y", "b_z"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV punti pista incompleto, mancano colonne: {sorted(missing)}")
        for row in reader:
            left_rows.append([float(row["a_x"]), float(row["a_y"]), float(row["a_z"])])
            right_rows.append([float(row["b_x"]), float(row["b_y"]), float(row["b_z"])])

    if not left_rows:
        raise ValueError(f"CSV punti pista vuoto: {path}")

    left = np.asarray(left_rows, dtype=float)
    right = np.asarray(right_rows, dtype=float)
    if "left_3d" in reconstruction and len(left) != len(reconstruction["left_3d"]):
        raise ValueError(
            "I punti corretti di viz_slope non corrispondono alla ricostruzione corrente: "
            f"{len(left)} nel CSV, {len(reconstruction['left_3d'])} nella NPZ.",
        )

    view = dict(reconstruction)
    view["left_3d"] = left
    view["right_3d"] = right
    view["center_3d"] = 0.5 * (left + right)
    return view


# Prepare reconstruction for display
def prepare_view_reconstruction(
    reconstruction: dict[str, np.ndarray],
    annotations: dict,
) -> tuple[dict[str, np.ndarray], dict | None]:
    try:
        bundle = build_homography_bundle(reconstruction, annotations)
        save_homography_outputs(bundle, reconstruction)
    except Exception as error:
        print(f"Omografia non calcolata: {error}")
        return reconstruction, None

    if not USE_HOMOGRAPHY_CORRECTED_POINTS:
        return reconstruction, bundle

    view = dict(reconstruction)
    view["left_3d"] = bundle["left_3d"]
    view["right_3d"] = bundle["right_3d"]
    view["center_3d"] = bundle["center_3d"]
    view["normal"] = bundle["normal"]
    view["track_width"] = bundle["track_width"]
    return view, bundle


# Describe reconstruction results
def reconstruction_summary(reconstruction: dict[str, np.ndarray], annotations: dict, homography_bundle: dict | None = None) -> str:
    left = reconstruction["left_3d"]
    right = reconstruction["right_3d"]
    width = float(np.ravel(reconstruction["track_width"])[0])
    normal = reconstruction["normal"]
    distances = np.linalg.norm(right - left, axis=1)
    normal_distances = (right - left) @ unit(normal, "normale")
    normal_angle = annotations.get("normal_axis_angle_degrees")
    plane_angle = annotations.get("plane_axis_angle_degrees")

    lines = [
        f"Punti per curva: {len(left)}",
        f"Track width relativo: {width:.6g}",
        f"Separazione normale media: {float(np.mean(normal_distances)):.6g}",
        f"Distanza euclidea media fra coppie: {float(np.mean(distances)):.6g}",
        f"Normale camera: [{normal[0]:.6g}, {normal[1]:.6g}, {normal[2]:.6g}]",
        "Camera: posizione [0, 0, 0], orientamento identita, asse ottico +Z.",
    ]
    if normal_angle is not None:
        lines.append(f"Angolo normale-asse ottico: {float(normal_angle):.3f} deg")
    if plane_angle is not None:
        lines.append(f"Angolo piano-asse ottico: {float(plane_angle):.3f} deg")
    if homography_bundle is not None:
        verification = homography_bundle["verification"]
        lines.append(f"Omografia reprojection mean/max: {verification['mean_reprojection_error_px']:.3e} / {verification['max_reprojection_error_px']:.3e} px")
        lines.append(f"Errore piano mean/max: {verification['mean_plane_error']:.3e} / {verification['max_plane_error']:.3e}")
        lines.append(f"Residuo traslazione fra curve medio: {verification['mean_translation_residual']:.6g}")
    return "\n".join(lines)


# Build curve line indices
def line_indices_for_curve(offset: int, count: int) -> list[list[int]]:
    return [[offset + idx, offset + idx + 1] for idx in range(count - 1)]


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


# Build camera frustum points
def camera_frustum_points(K: np.ndarray, image_size: tuple[int, int], scene_points: np.ndarray) -> np.ndarray:
    width, height = image_size
    span = float(np.max(np.ptp(scene_points, axis=0)))
    depth = max(0.5, span * CAMERA_FRUSTUM_SCALE)
    inv_K = np.linalg.inv(K)
    corners = np.array(
        [
            [0.0, 0.0, 1.0],
            [float(width), 0.0, 1.0],
            [float(width), float(height), 1.0],
            [0.0, float(height), 1.0],
        ],
        dtype=float,
    )
    rays = (inv_K @ corners.T).T
    return rays * (depth / rays[:, 2:3])


# Build camera axis points
def camera_axis_points(scene_points: np.ndarray) -> np.ndarray:
    span = float(np.max(np.ptp(scene_points, axis=0)))
    scale = max(0.5, span * CAMERA_AXIS_SCALE)
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [scale, 0.0, 0.0],
            [0.0, scale, 0.0],
            [0.0, 0.0, scale],
        ],
        dtype=float,
    )


# Draw camera plot geometry
def draw_camera_matplotlib(ax, K: np.ndarray, image_size: tuple[int, int], scene_points: np.ndarray) -> np.ndarray:
    corners = camera_frustum_points(K, image_size, scene_points)
    center = np.zeros(3, dtype=float)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

    for corner in corners:
        segment = np.vstack([center, corner])
        ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color=CAMERA_COLOR, linewidth=1.0, alpha=0.9)
    for first, second in edges:
        segment = np.vstack([corners[first], corners[second]])
        ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color=CAMERA_COLOR, linewidth=1.0, alpha=0.9)

    axis_points = camera_axis_points(scene_points)
    ax.scatter([0.0], [0.0], [0.0], color=CAMERA_COLOR, s=28, label="camera")
    ax.text(0.0, 0.0, 0.0, "camera", color=CAMERA_COLOR)
    ax.quiver(0.0, 0.0, 0.0, axis_points[1, 0], axis_points[1, 1], axis_points[1, 2], color=CAMERA_X_COLOR, linewidth=2.0)
    ax.quiver(0.0, 0.0, 0.0, axis_points[2, 0], axis_points[2, 1], axis_points[2, 2], color=CAMERA_Y_COLOR, linewidth=2.0)
    ax.quiver(0.0, 0.0, 0.0, axis_points[3, 0], axis_points[3, 1], axis_points[3, 2], color=CAMERA_Z_COLOR, linewidth=2.0)
    ax.text(*axis_points[1], "X", color=CAMERA_X_COLOR)
    ax.text(*axis_points[2], "Y", color=CAMERA_Y_COLOR)
    ax.text(*axis_points[3], "Z", color=CAMERA_Z_COLOR)
    return np.vstack([center, corners, axis_points])


# Draw plane axes in a plot
def draw_plane_basis_matplotlib(ax, bundle: dict) -> np.ndarray:
    origin = bundle["origin_a"]
    scale = PLANE_BASIS_SCALE
    u_end = origin + scale * bundle["e1"]
    v_end = origin + scale * bundle["e2"]
    ax.quiver(origin[0], origin[1], origin[2], u_end[0] - origin[0], u_end[1] - origin[1], u_end[2] - origin[2], color=PLANE_U_COLOR, linewidth=2.0, label="plane u")
    ax.quiver(origin[0], origin[1], origin[2], v_end[0] - origin[0], v_end[1] - origin[1], v_end[2] - origin[2], color=PLANE_V_COLOR, linewidth=2.0, label="plane v")
    ax.text(*u_end, "u", color=PLANE_U_COLOR)
    ax.text(*v_end, "v", color=PLANE_V_COLOR)
    return np.vstack([origin, u_end, v_end])


# Display reconstruction with plots
def show_with_matplotlib(reconstruction: dict[str, np.ndarray], annotations: dict, homography_bundle: dict | None) -> None:
    import matplotlib.pyplot as plt

    left = reconstruction["left_3d"]
    right = reconstruction["right_3d"]
    center = reconstruction["center_3d"]
    normal = reconstruction["normal"]
    all_points = np.vstack([left, right, center])

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    manager = getattr(fig.canvas, "manager", None)
    if manager is not None and hasattr(manager, "set_window_title"):
        manager.set_window_title(WINDOW_TITLE)

    ax.plot(left[:, 0], left[:, 1], left[:, 2], color=LEFT_COLOR, linewidth=MATPLOTLIB_LINE_WIDTH, label="polyline A")
    ax.plot(right[:, 0], right[:, 1], right[:, 2], color=RIGHT_COLOR, linewidth=MATPLOTLIB_LINE_WIDTH, label="polyline B")

    if SHOW_CENTERLINE:
        ax.plot(center[:, 0], center[:, 1], center[:, 2], "--", color=CENTER_COLOR, linewidth=1.4, label="center")

    if SHOW_RULINGS:
        stride = max(1, RULING_STRIDE)
        for idx in range(0, len(left), stride):
            segment = np.vstack([left[idx], right[idx]])
            ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color=RULING_COLOR, linewidth=0.8, alpha=0.75)

    if SHOW_POINTS:
        ax.scatter(left[:, 0], left[:, 1], left[:, 2], color=LEFT_COLOR, s=POINT_SIZE)
        ax.scatter(right[:, 0], right[:, 1], right[:, 2], color=RIGHT_COLOR, s=POINT_SIZE)

    if SHOW_NORMAL:
        origin = np.mean(center, axis=0)
        end = origin + NORMAL_SCALE * normal / max(float(np.linalg.norm(normal)), 1e-12)
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            end[0] - origin[0],
            end[1] - origin[1],
            end[2] - origin[2],
            color=NORMAL_COLOR,
            linewidth=2.5,
            label="normal",
        )
        all_points = np.vstack([all_points, origin, end])

    if SHOW_CAMERA:
        try:
            K = camera_matrix_from_annotations(annotations)
            image_size = image_size_from_annotations(annotations, K)
            camera_points = draw_camera_matplotlib(ax, K, image_size, all_points)
            all_points = np.vstack([all_points, camera_points])
        except Exception as error:
            print(f"Camera non disegnata: {error}")

    if SHOW_PLANE_BASES and homography_bundle is not None:
        basis_points = draw_plane_basis_matplotlib(ax, homography_bundle)
        all_points = np.vstack([all_points, basis_points])

    ax.set_xlabel("X camera")
    ax.set_ylabel("Y camera")
    ax.set_zlabel("Z camera")
    ax.set_title("Slope 3D reconstruction")
    ax.legend(loc="upper right")
    set_axes_equal(ax, all_points)
    ax.view_init(elev=22, azim=-62)
    fig.text(0.02, 0.02, "Mouse: ruota, pan e zoom dalla toolbar di Matplotlib.", fontsize=9)
    fig.tight_layout()

    print(reconstruction_summary(reconstruction, annotations, homography_bundle))
    print("Viewer Matplotlib aperto. Usa mouse e toolbar per ruotare, zoomare e spostarti.")
    plt.show()


# Build lines for interactive display
def open3d_line_set(o3d, points: np.ndarray, lines: list[list[int]], colors: list[tuple[float, float, float]]):
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    line_set.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=float))
    return line_set


# Build camera interactive geometry
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


# Build plane interactive geometry
def plane_basis_open3d_geometry(o3d, bundle: dict):
    origin = bundle["origin_a"]
    points = np.vstack(
        [
            origin,
            origin + PLANE_BASIS_SCALE * bundle["e1"],
            origin + PLANE_BASIS_SCALE * bundle["e2"],
        ],
    )
    lines = [[0, 1], [0, 2]]
    colors = [PLANE_U_COLOR, PLANE_V_COLOR]
    return open3d_line_set(o3d, points, lines, colors)


# Display reconstruction interactively
def show_with_open3d(reconstruction: dict[str, np.ndarray], annotations: dict, homography_bundle: dict | None) -> None:
    try:
        import open3d as o3d
    except ImportError as error:
        raise RuntimeError("Open3D non installato: usa venv/bin/python -m pip install open3d") from error

    left = reconstruction["left_3d"]
    right = reconstruction["right_3d"]
    center = reconstruction["center_3d"]
    normal = reconstruction["normal"]
    count = len(left)

    points_blocks = [left, right]
    lines = []
    colors = []

    lines.extend(line_indices_for_curve(0, count))
    colors.extend([LEFT_COLOR] * (count - 1))
    lines.extend(line_indices_for_curve(count, count))
    colors.extend([RIGHT_COLOR] * (count - 1))

    if SHOW_CENTERLINE:
        center_offset = sum(len(block) for block in points_blocks)
        points_blocks.append(center)
        lines.extend(line_indices_for_curve(center_offset, count))
        colors.extend([CENTER_COLOR] * (count - 1))

    if SHOW_RULINGS:
        stride = max(1, RULING_STRIDE)
        for idx in range(0, count, stride):
            lines.append([idx, count + idx])
            colors.append(RULING_COLOR)

    if SHOW_NORMAL:
        normal_offset = sum(len(block) for block in points_blocks)
        origin = np.mean(center, axis=0)
        end = origin + NORMAL_SCALE * normal / max(float(np.linalg.norm(normal)), 1e-12)
        points_blocks.append(np.vstack([origin, end]))
        lines.append([normal_offset, normal_offset + 1])
        colors.append(NORMAL_COLOR)

    scene_points = np.vstack([left, right, center])
    geometries = [open3d_line_set(o3d, np.vstack(points_blocks), lines, colors)]

    if SHOW_POINTS:
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(np.vstack([left, right]))
        point_colors = np.vstack([
            np.tile(np.asarray(LEFT_COLOR), (count, 1)),
            np.tile(np.asarray(RIGHT_COLOR), (count, 1)),
        ])
        point_cloud.colors = o3d.utility.Vector3dVector(point_colors)
        geometries.append(point_cloud)

    if SHOW_CAMERA:
        try:
            K = camera_matrix_from_annotations(annotations)
            image_size = image_size_from_annotations(annotations, K)
            geometries.append(camera_open3d_geometry(o3d, K, image_size, scene_points))
        except Exception as error:
            print(f"Camera non disegnata: {error}")

    if SHOW_PLANE_BASES and homography_bundle is not None:
        geometries.append(plane_basis_open3d_geometry(o3d, homography_bundle))

    if SHOW_AXES:
        span = float(np.max(np.ptp(scene_points, axis=0)))
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=max(0.5, span * 0.2), origin=[0.0, 0.0, 0.0])
        geometries.append(axes)

    print(reconstruction_summary(reconstruction, annotations, homography_bundle))
    print("Viewer Open3D aperto. Mouse sinistro ruota, rotella zoom, mouse destro o shift+sinistro sposta.")

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(window_name=WINDOW_TITLE, width=1200, height=850)
    for geometry in geometries:
        visualizer.add_geometry(geometry)

    options = visualizer.get_render_option()
    options.background_color = np.asarray(BACKGROUND_COLOR, dtype=float)
    options.point_size = POINT_SIZE
    options.line_width = 2.0

    visualizer.run()
    visualizer.destroy_window()


# Run the visualization workflow
def main() -> int:
    reconstruction = load_reconstruction(RECONSTRUCTION_NPZ_PATH)
    annotations = load_annotations(ANNOTATIONS_JSON_PATH)
    view_reconstruction, homography_bundle = prepare_view_reconstruction(reconstruction, annotations)

    backend = VIEWER_BACKEND.lower().strip()
    if backend not in {"auto", "open3d", "matplotlib"}:
        raise ValueError("VIEWER_BACKEND deve essere 'auto', 'open3d' o 'matplotlib'.")

    if backend in {"auto", "open3d"}:
        try:
            show_with_open3d(view_reconstruction, annotations, homography_bundle)
            return 0
        except RuntimeError as error:
            if backend == "open3d":
                raise
            print(f"{error}. Uso Matplotlib come fallback.")

    show_with_matplotlib(view_reconstruction, annotations, homography_bundle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
