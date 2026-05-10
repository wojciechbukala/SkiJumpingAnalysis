from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.track_geometry.calibration import CalibrationError, estimate_k_from_frame


FRAME_PATH = Path("extracted_frames/G_30out_frame_00004.jpg")
MAX_LINES = 120
SELECTED_LINE_GROUPS = [
    [82, 79, 13, 3,22],
    [33,81,37,42,11,1,40,7,10],      
    [70,90,55,34],   
]


SELECTED_LINE_IDS: list[int] | None = None
PREVIEW_PATH = Path("outputGeometry/calibration_lines_preview.jpg")
EDGES_PATH = Path("outputGeometry/calibration_edges.jpg")
LINES_CSV_PATH = Path("outputGeometry/calibration_lines.csv")
OUTPUT_PATH = Path("outputGeometry/calibration_K.json")
SHOW_WINDOW = False
PREVIEW_ONLY = False

LINE_SEARCH_TOP_RATIO = 0.6

USE_LSD_DETECTOR = True
USE_HOUGH_DETECTOR = False
USE_VERTICAL_CLOSE_FOR_HOUGH = False
VERTICAL_CLOSE_KERNEL_WIDTH = 3
VERTICAL_CLOSE_KERNEL_HEIGHT = 25

LSD_MIN_LINE_LENGTH_PX = 25.0
LSD_MAX_LINE_WIDTH_PX = 6.0
LSD_MIN_EDGE_SUPPORT_RATIO = 0.12
EDGE_SUPPORT_RADIUS_PX = 1

HOUGH_ATTEMPTS = [
    (90, 0.025),
    (65, 0.035),
    (45, 0.045),
]
HOUGH_MIN_LINE_LENGTH_PX = 40
HOUGH_MIN_LINE_LENGTH_RATIO = 0.0
HOUGH_MAX_LINE_GAP_PX = 60

FILTER_MIN_LINE_LENGTH_PX = 20.0
FILTER_MIN_LINE_LENGTH_RATIO = 0.0

DEDUP_ANGLE_TOLERANCE_DEG = 1.5
DEDUP_DISTANCE_TOLERANCE_PX = 8.0
DEDUP_MIDPOINT_TOLERANCE_PX = 90.0


def preprocess_for_lines(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Enhance image gradients before Canny so Hough sees structural borders.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)

    sobel_x = cv2.Sobel(contrast, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(contrast, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobel_x, sobel_y)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    enhanced = cv2.addWeighted(contrast, 0.55, magnitude, 0.45, 0.0)
    median = float(np.median(enhanced))
    low = int(max(25, 0.66 * median))
    high = int(min(255, max(low + 35, 1.33 * median)))

    edges = cv2.Canny(enhanced, low, high, L2gradient=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    return enhanced, edges


def detect_lines(
    enhanced: np.ndarray,
    edges: np.ndarray,
    max_lines: int,
) -> list[tuple[int, int, int, int]]:
    roi_y0 = get_roi_y0(edges.shape[0])
    roi_enhanced = enhanced[roi_y0:, :]
    roi_edges = edges[roi_y0:, :]
    height, width = roi_edges.shape[:2]
    min_dim = min(width, height)
    min_keep_length = max(FILTER_MIN_LINE_LENGTH_PX, min_dim * FILTER_MIN_LINE_LENGTH_RATIO)

    candidate_lines = []

    if USE_LSD_DETECTOR:
        candidate_lines.extend(detect_lsd_lines(roi_enhanced, roi_edges, roi_y0, min_keep_length))

    if USE_HOUGH_DETECTOR:
        candidate_lines.extend(detect_hough_lines(roi_edges, roi_y0, min_dim, min_keep_length))

    return select_lines(candidate_lines, max_lines=max_lines)


def get_roi_y0(height: int) -> int:
    roi_y0 = int(height * LINE_SEARCH_TOP_RATIO)
    return max(0, min(height - 1, roi_y0))


def detect_lsd_lines(
    roi_enhanced: np.ndarray,
    roi_edges: np.ndarray,
    roi_y0: int,
    min_keep_length: float,
) -> list[tuple[float, tuple[int, int, int, int]]]:
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    detected = detector.detect(roi_enhanced)
    lines = detected[0]
    widths = detected[1]

    if lines is None:
        return []

    candidate_lines = []
    for idx, raw_line in enumerate(lines[:, 0, :]):
        width = float(widths[idx][0]) if widths is not None else 0.0
        if width > LSD_MAX_LINE_WIDTH_PX:
            continue

        x1, y1, x2, y2 = [float(value) for value in raw_line]
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < max(min_keep_length, LSD_MIN_LINE_LENGTH_PX):
            continue

        support = edge_support_ratio(roi_edges, x1, y1, x2, y2)
        if support < LSD_MIN_EDGE_SUPPORT_RATIO:
            continue

        candidate_lines.append((length, rounded_line(x1, y1 + roi_y0, x2, y2 + roi_y0)))

    return candidate_lines


def detect_hough_lines(
    roi_edges: np.ndarray,
    roi_y0: int,
    min_dim: int,
    min_keep_length: float,
) -> list[tuple[float, tuple[int, int, int, int]]]:
    search_edges = roi_edges.copy()

    if USE_VERTICAL_CLOSE_FOR_HOUGH:
        vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (VERTICAL_CLOSE_KERNEL_WIDTH, VERTICAL_CLOSE_KERNEL_HEIGHT),
        )
        vertical_edges = cv2.morphologyEx(search_edges, cv2.MORPH_CLOSE, vertical_kernel, iterations=1)
        search_edges = cv2.bitwise_or(search_edges, vertical_edges)

    min_hough_length = max(HOUGH_MIN_LINE_LENGTH_PX, int(min_dim * HOUGH_MIN_LINE_LENGTH_RATIO))
    candidate_lines = []

    for threshold, max_gap_ratio in HOUGH_ATTEMPTS:
        segments = cv2.HoughLinesP(
            search_edges,
            rho=1,
            theta=np.pi / 180.0,
            threshold=threshold,
            minLineLength=min_hough_length,
            maxLineGap=max(HOUGH_MAX_LINE_GAP_PX, int(min_dim * max_gap_ratio)),
        )
        if segments is None:
            continue

        for raw_segment in segments[:, 0, :]:
            x1, y1, x2, y2 = [int(value) for value in raw_segment]
            y1 += roi_y0
            y2 += roi_y0
            length = float(np.hypot(x2 - x1, y2 - y1))
            if length < min_keep_length:
                continue
            candidate_lines.append((length, (x1, y1, x2, y2)))

    return candidate_lines


def build_search_edges(edges: np.ndarray) -> tuple[int, np.ndarray]:
    height = edges.shape[0]
    roi_y0 = get_roi_y0(height)
    search_edges = edges[roi_y0:, :].copy()

    if not USE_VERTICAL_CLOSE_FOR_HOUGH:
        return roi_y0, search_edges

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (VERTICAL_CLOSE_KERNEL_WIDTH, VERTICAL_CLOSE_KERNEL_HEIGHT),
    )
    vertical_edges = cv2.morphologyEx(search_edges, cv2.MORPH_CLOSE, vertical_kernel, iterations=1)
    search_edges = cv2.bitwise_or(search_edges, vertical_edges)
    return roi_y0, search_edges


def edge_support_ratio(edges: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> float:
    length = float(np.hypot(x2 - x1, y2 - y1))
    samples = max(8, int(length))
    xs = np.linspace(x1, x2, samples)
    ys = np.linspace(y1, y2, samples)
    hits = 0

    for x_float, y_float in zip(xs, ys):
        x = int(round(x_float))
        y = int(round(y_float))
        x0 = max(0, x - EDGE_SUPPORT_RADIUS_PX)
        x1_box = min(edges.shape[1], x + EDGE_SUPPORT_RADIUS_PX + 1)
        y0 = max(0, y - EDGE_SUPPORT_RADIUS_PX)
        y1_box = min(edges.shape[0], y + EDGE_SUPPORT_RADIUS_PX + 1)
        if np.any(edges[y0:y1_box, x0:x1_box] > 0):
            hits += 1

    return hits / float(samples)


def rounded_line(x1: float, y1: float, x2: float, y2: float) -> tuple[int, int, int, int]:
    return (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))


def select_lines(
    candidate_lines: list[tuple[float, tuple[int, int, int, int]]],
    max_lines: int,
) -> list[tuple[int, int, int, int]]:
    candidate_lines = sorted(candidate_lines, key=lambda item: item[0], reverse=True)
    selected = []
    used = set()

    for _, line in candidate_lines:
        if len(selected) >= max_lines:
            break
        key = tuple(line)
        if key in used:
            continue
        if any(lines_are_too_similar(line, existing) for existing in selected):
            continue
        selected.append(line)
        used.add(key)

    return selected[:max_lines]


def lines_are_too_similar(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> bool:
    angle_a = segment_angle(first)
    angle_b = segment_angle(second)
    angle_delta = abs(angle_a - angle_b)
    angle_delta = min(angle_delta, 180.0 - angle_delta)
    if angle_delta > DEDUP_ANGLE_TOLERANCE_DEG:
        return False

    midpoint_a = segment_midpoint(first)
    midpoint_b = segment_midpoint(second)
    if float(np.linalg.norm(midpoint_a - midpoint_b)) > DEDUP_MIDPOINT_TOLERANCE_PX:
        return False

    distance_a = point_to_line_distance(midpoint_a, second)
    distance_b = point_to_line_distance(midpoint_b, first)
    return max(distance_a, distance_b) <= DEDUP_DISTANCE_TOLERANCE_PX


def segment_angle(segment: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = segment
    return float(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0)


def segment_midpoint(segment: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = segment
    return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=float)


def point_to_line_distance(point: np.ndarray, segment: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = [float(value) for value in segment]
    dx = x2 - x1
    dy = y2 - y1
    denom = float(np.hypot(dx, dy))
    if denom < 1e-12:
        return float("inf")
    return abs(dy * point[0] - dx * point[1] + x2 * y1 - y2 * x1) / denom


def draw_numbered_lines(
    frame: np.ndarray,
    lines: list[tuple[int, int, int, int]],
) -> np.ndarray:
    preview = frame.copy()
    for idx, (x1, y1, x2, y2) in enumerate(lines):
        color = (
            int(40 + (37 * idx) % 215),
            int(60 + (83 * idx) % 195),
            int(90 + (131 * idx) % 165),
        )
        midpoint = ((x1 + x2) // 2, (y1 + y2) // 2)
        cv2.line(preview, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        cv2.circle(preview, midpoint, 4, (0, 0, 255), -1)
        cv2.putText(
            preview,
            str(idx),
            (midpoint[0] + 5, midpoint[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return preview


def choose_line_groups(
    lines: list[tuple[int, int, int, int]],
    provided_groups: list[list[int]] | None,
    legacy_ids: list[int] | None,
) -> list[list[int]]:
    # Select three groups. Each group contains at least two image lines that are
    # parallel in the scene and converge to the same vanishing point.
    if provided_groups is not None:
        groups = [[int(value) for value in group] for group in provided_groups]
    elif legacy_ids is not None:
        selected = [int(value) for value in legacy_ids]
        if len(selected) != 6:
            raise ValueError(f"SELECTED_LINE_IDS deve contenere 6 ID, ricevuti {len(selected)}.")
        groups = [selected[0:2], selected[2:4], selected[4:6]]
    else:
        raw = input(
            "Inserisci 3 gruppi separati da ';'. "
            "Esempio: 0,4,12; 3,9,18; 7,22,31: "
        )
        groups = [
            [int(value) for value in chunk.replace(",", " ").split()]
            for chunk in raw.split(";")
            if chunk.strip()
        ]

    if len(groups) != 3:
        raise ValueError(f"Servono esattamente 3 gruppi di linee, ricevuti {len(groups)}.")

    for group_idx, group in enumerate(groups):
        if len(group) < 2:
            raise ValueError(f"Il gruppo {group_idx} ha meno di 2 linee: {group}.")

    all_ids = [idx for group in groups for idx in group]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("Ogni linea deve appartenere a un solo gruppo.")

    invalid = [idx for idx in all_ids if idx < 0 or idx >= len(lines)]
    if invalid:
        raise ValueError(f"ID linea non validi: {invalid}. Linee disponibili: 0..{len(lines) - 1}.")

    return groups


def line_from_segment(segment: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = [float(value) for value in segment]
    line = np.array([y1 - y2, x2 - x1, x1 * y2 - x2 * y1], dtype=float)
    norm = float(np.linalg.norm(line[:2]))
    if norm < 1e-12:
        raise ValueError("Linea degenerata.")
    return line / norm


def vanishing_point_from_line_group(segments: list[tuple[int, int, int, int]]) -> list[float]:
    line_matrix = np.vstack([line_from_segment(segment) for segment in segments])
    _, _, vh = np.linalg.svd(line_matrix)
    point = vh[-1, :]

    if abs(float(point[2])) < 1e-9:
        raise CalibrationError("Il gruppo di linee produce un punto di fuga quasi all'infinito.")

    point = point / point[2]
    if not np.all(np.isfinite(point)):
        raise CalibrationError("Punto di fuga non finito.")

    return [float(point[0]), float(point[1])]


def vanishing_points_from_line_groups(
    lines: list[tuple[int, int, int, int]],
    selected_groups: list[list[int]],
) -> list[list[float]]:
    return [
        vanishing_point_from_line_group([lines[idx] for idx in group])
        for group in selected_groups
    ]


def print_vanishing_points(selected_groups: list[list[int]], vanishing_points: list[list[float]]) -> None:
    print("Punti di fuga stimati:")
    for group_idx, (group, point) in enumerate(zip(selected_groups, vanishing_points), start=1):
        print(f"  Gruppo {group_idx} linee {group}: x={point[0]:.6f}, y={point[1]:.6f}")


def estimate_from_line_groups(
    frame: np.ndarray,
    lines: list[tuple[int, int, int, int]],
    selected_groups: list[list[int]],
) -> tuple[np.ndarray, list[list[float]], list[list[list[float]]]]:
    vanishing_points = vanishing_points_from_line_groups(lines, selected_groups)
    print_vanishing_points(selected_groups, vanishing_points)

    pairs = [
        [vanishing_points[0], vanishing_points[1]],
        [vanishing_points[0], vanishing_points[2]],
        [vanishing_points[1], vanishing_points[2]],
    ]
    K = estimate_k_from_frame(frame, orthogonal_pairs=pairs)
    return K, vanishing_points, pairs


def main() -> int:
    frame = cv2.imread(str(FRAME_PATH))
    if frame is None:
        raise FileNotFoundError(f"Frame non trovato o non leggibile: {FRAME_PATH}")

    enhanced, edges = preprocess_for_lines(frame)
    lines = detect_lines(enhanced, edges, max_lines=MAX_LINES)
    if len(lines) < 6:
        raise RuntimeError(f"Il detector ha trovato solo {len(lines)} linee utili; ne servono almeno 6.")

    PREVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    preview = draw_numbered_lines(frame, lines)
    search_edges_preview = np.zeros_like(edges)
    search_y0, search_edges = build_search_edges(edges)
    search_edges_preview[search_y0:, :] = search_edges
    cv2.imwrite(str(PREVIEW_PATH), preview)
    cv2.imwrite(str(EDGES_PATH), search_edges_preview)
    save_lines_csv(LINES_CSV_PATH, lines)

    print(f"Frame usato: {FRAME_PATH}")
    print(f"Linee rilevate: {len(lines)}")
    print(f"Preview linee numerate: {PREVIEW_PATH}")
    print(f"Edges filtrati: {EDGES_PATH}")
    print(f"Report linee: {LINES_CSV_PATH}")

    if PREVIEW_ONLY:
        if SELECTED_LINE_GROUPS is not None or SELECTED_LINE_IDS is not None:
            selected_groups = choose_line_groups(lines, SELECTED_LINE_GROUPS, SELECTED_LINE_IDS)
            vanishing_points = vanishing_points_from_line_groups(lines, selected_groups)
            print_vanishing_points(selected_groups, vanishing_points)
        print("Preview generata. Imposta PREVIEW_ONLY = False e SELECTED_LINE_GROUPS = [[...], [...], [...]] per stimare K.")
        return 0

    if SHOW_WINDOW:
        try:
            cv2.imshow("Hough lines - scegli 6 ID dal terminale", preview)
            cv2.waitKey(1)
        except cv2.error:
            print("OpenCV GUI non disponibile: usa il file preview salvato.")

    selected_groups = choose_line_groups(lines, SELECTED_LINE_GROUPS, SELECTED_LINE_IDS)
    try:
        K, vanishing_points, pairs = estimate_from_line_groups(frame, lines, selected_groups)
    except CalibrationError as error:
        print(f"Calibrazione non riuscita: {error}")
        return 1

    payload = {
        "frame": str(FRAME_PATH),
        "selected_line_groups": selected_groups,
        "vanishing_points": vanishing_points,
        "orthogonal_pairs": pairs,
        "K": K.tolist(),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    print("Estimated Intrinsic Matrix K:")
    print(K)
    print(f"Risultato salvato in: {OUTPUT_PATH}")

    if SHOW_WINDOW:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0


def save_lines_csv(path: Path, lines: list[tuple[int, int, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("id,x1,y1,x2,y2,length,angle_deg\n")
        for idx, (x1, y1, x2, y2) in enumerate(lines):
            dx = x2 - x1
            dy = y2 - y1
            length = float(np.hypot(dx, dy))
            angle = float(np.degrees(np.arctan2(dy, dx)) % 180.0)
            handle.write(f"{idx},{x1},{y1},{x2},{y2},{length:.3f},{angle:.3f}\n")


if __name__ == "__main__":
    raise SystemExit(main())
