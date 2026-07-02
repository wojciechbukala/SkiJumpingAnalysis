# Camera calibration workflow
# Step one loads FRAME_PATH and extracts enhanced edges and candidate image lines -- gray,gaussianblur,clahe(enanched contrast),gradient intensity with sobel, canny to extract edges(thresholds low and high function of image median luminance), LSD line detector on enhanced image and edges with a max line limit of MAX_LINES.
# Step two saves numbered line previews edge previews and a CSV report of detected lines
# Step three chooses three line groups either from SELECTED_LINE_GROUPS SELECTED_LINE_IDS or terminal input
# Step four estimates one vanishing point per selected line group
# Step five builds the three orthogonal vanishing point pairs between those groups
# Step six estimates K either from a fixed principal point or from equal length segment constraints
# Step seven saves frame metadata selected groups vanishing points pairs and K to OUTPUT_PATH
# Option PREVIEW_ONLY stops after previews and selected vanishing points
# Option SHOW_WINDOW displays OpenCV windows when available
# Option DRAW_EQUAL_LENGTH_SEGMENTS enables interactive equal segment selection
# Option PRINCIPAL_POINT fixes the principal point when known
# Option ESTIMATE_PRINCIPAL_POINT requires equal length segment input when PRINCIPAL_POINT is absent
# Option EQUAL_LENGTH_PLANE_NORMAL_GROUP selects which vanishing group is normal to the equal segment plane
# Formula image line from segment is [y1-y2 x2-x1 x1*y2-x2*y1]
# Formula vanishing point is the last singular vector of the selected line matrix
# Formula selected orthogonal pairs are vp1 vp2 vp1 vp3 and vp2 vp3
# Formula intrinsic estimation is delegated to calibration.py and equal length constraints when needed

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.track_geometry.calibration import (
    CalibrationError,
    estimate_k_from_orthogonal_vanishing_pairs,
    estimate_k_from_orthogonal_vanishing_pairs_and_equal_lengths,
)
from src.track_geometry.line_extraction import (
    build_search_edges,
    detect_lines,
    preprocess_for_lines,
)
from src.track_geometry.tool_line import (
    choose_equal_length_segments,
    choose_line_groups,
    draw_equal_length_segments_preview,
    draw_numbered_lines,
)


FRAME_PATH = Path("extracted_frames/G_30out_frame_00004.jpg")
MAX_LINES = 120
# THE LAST GROUP CAN BE USED TO ESTIMATE THE PRINCIPAL POINT IF ESTIMATE_PRINCIPAL_POINT=True
# SELECT THE THIRD GROUP TO BE PERPENDICULAR TO THE PLANE OF THE EQUAL LENGTH SEGMENTS IF ESTIMATE_PRINCIPAL_POINT=True
SELECTED_LINE_GROUPS = [
    [82, 79, 13, 3,22],
    [33,81,37,42,11,1,40,7,10],
    [70,90,55,34],
]
EQUAL_LENGTH_SEGMENTS: list[tuple[tuple[float, float], tuple[float, float]]] | None = None
EQUAL_LENGTH_PLANE_NORMAL_GROUP = 3


SELECTED_LINE_IDS: list[int] | None = None
PREVIEW_PATH = Path("outputGeometry/calibration_lines_preview.jpg")
EDGES_PATH = Path("outputGeometry/calibration_edges.jpg")
LINES_CSV_PATH = Path("outputGeometry/calibration_lines.csv")
EQUAL_LENGTH_PREVIEW_PATH = Path("outputGeometry/calibration_equal_length_segments.jpg")
OUTPUT_PATH = Path("outputGeometry/calibration_K.json")
SHOW_WINDOW = False
DRAW_EQUAL_LENGTH_SEGMENTS = True
PREVIEW_ONLY = False
PRINCIPAL_POINT: tuple[float, float] | None = None
ESTIMATE_PRINCIPAL_POINT = True


# Convert a segment to a line
def line_from_segment(segment: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = [float(value) for value in segment]
    line = np.array([y1 - y2, x2 - x1, x1 * y2 - x2 * y1], dtype=float)
    norm = float(np.linalg.norm(line[:2]))
    if norm < 1e-12:
        raise ValueError("Linea degenerata.")
    return line / norm


# Estimate one vanishing point
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


# Estimate vanishing points from groups
def vanishing_points_from_line_groups(
    lines: list[tuple[int, int, int, int]],
    selected_groups: list[list[int]],
) -> list[list[float]]:
    return [
        vanishing_point_from_line_group([lines[idx] for idx in group])
        for group in selected_groups
    ]


# Print estimated vanishing points
def print_vanishing_points(selected_groups: list[list[int]], vanishing_points: list[list[float]]) -> None:
    print("Punti di fuga stimati:")
    for group_idx, (group, point) in enumerate(zip(selected_groups, vanishing_points), start=1):
        print(f"  Gruppo {group_idx} linee {group}: x={point[0]:.6f}, y={point[1]:.6f}")


# Estimate calibration from selected lines
def estimate_from_line_groups(
    frame: np.ndarray,
    lines: list[tuple[int, int, int, int]],
    selected_groups: list[list[int]],
    equal_length_segments: list[tuple[tuple[float, float], tuple[float, float]]] | None,
) -> tuple[np.ndarray, list[list[float]], list[list[list[float]]]]:
    vanishing_points = vanishing_points_from_line_groups(lines, selected_groups)
    print_vanishing_points(selected_groups, vanishing_points)

    pairs = [
        [vanishing_points[0], vanishing_points[1]],
        [vanishing_points[0], vanishing_points[2]],
        [vanishing_points[1], vanishing_points[2]],
    ]

    if ESTIMATE_PRINCIPAL_POINT and PRINCIPAL_POINT is None:
        if equal_length_segments is None:
            raise CalibrationError(
                "ESTIMATE_PRINCIPAL_POINT=True richiede due segmenti di uguale lunghezza reale "
                "oppure un altro vincolo metrico indipendente.",
            )
        normal_group_index = int(EQUAL_LENGTH_PLANE_NORMAL_GROUP) - 1
        if normal_group_index < 0 or normal_group_index >= len(vanishing_points):
            raise ValueError("EQUAL_LENGTH_PLANE_NORMAL_GROUP deve essere 1, 2 o 3.")
        K = estimate_k_from_orthogonal_vanishing_pairs_and_equal_lengths(
            pairs,
            equal_length_segments,
            vanishing_points[normal_group_index],
            image_shape=frame.shape[:2],
        )
        return K, vanishing_points, pairs

    K = estimate_k_from_orthogonal_vanishing_pairs(
        pairs,
        image_shape=frame.shape[:2],
        principal_point=PRINCIPAL_POINT,
    )
    return K, vanishing_points, pairs


# Run the calibration workflow
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
    equal_length_segments = None
    if ESTIMATE_PRINCIPAL_POINT and PRINCIPAL_POINT is None:
        equal_length_segments = choose_equal_length_segments(
            frame,
            EQUAL_LENGTH_SEGMENTS,
            draw_segments=DRAW_EQUAL_LENGTH_SEGMENTS,
            show_window=SHOW_WINDOW,
        )
        equal_length_preview = draw_equal_length_segments_preview(frame, equal_length_segments)
        cv2.imwrite(str(EQUAL_LENGTH_PREVIEW_PATH), equal_length_preview)
        print(f"Segmenti di uguale lunghezza reale: {equal_length_segments}")
        print(f"Normale del piano dei segmenti: gruppo {EQUAL_LENGTH_PLANE_NORMAL_GROUP}")
        print(f"Preview segmenti uguali: {EQUAL_LENGTH_PREVIEW_PATH}")

    try:
        K, vanishing_points, pairs = estimate_from_line_groups(
            frame,
            lines,
            selected_groups,
            equal_length_segments,
        )
    except CalibrationError as error:
        print(f"Calibrazione non riuscita: {error}")
        return 1

    payload = {
        "frame": str(FRAME_PATH),
        "selected_line_groups": selected_groups,
        "equal_length_segments": equal_length_segments,
        "equal_length_plane_normal_group": (
            EQUAL_LENGTH_PLANE_NORMAL_GROUP if equal_length_segments is not None else None
        ),
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


# Save detected lines to a table
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
