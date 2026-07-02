# Line selection and drawing tools
# Step one ranks candidate lines by detector score or segment length
# Step two removes exact duplicate segments and near duplicate segments
# Step three draws numbered line previews and equal length segment previews
# Step four parses or collects selected line groups and equal length segments
# Option provided_groups uses explicit line groups without prompting
# Option legacy_ids accepts six selected line ids and splits them into three pairs
# Option draw_segments opens the interactive equal segment selector when possible
# Option show_window also enables interactive selection behavior
# Formula line angle is atan2(y2-y1 x2-x1) in degrees modulo one hundred eighty
# Formula midpoint is ((x1+x2)/2 (y1+y2)/2)
# Formula point to line distance is abs(dy*x - dx*y + x2*y1 - y2*x1) / sqrt(dx^2 + dy^2)

from __future__ import annotations

import cv2
import numpy as np


DEDUP_ANGLE_TOLERANCE_DEG = 1.5
DEDUP_DISTANCE_TOLERANCE_PX = 8.0
DEDUP_MIDPOINT_TOLERANCE_PX = 90.0


# Keep distinct detected lines
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


# Compare two detected lines
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


# Compute a segment angle
def segment_angle(segment: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = segment
    return float(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0)


# Compute a segment midpoint
def segment_midpoint(segment: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = segment
    return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=float)


# Measure distance from a point to a line
def point_to_line_distance(point: np.ndarray, segment: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = [float(value) for value in segment]
    dx = x2 - x1
    dy = y2 - y1
    denom = float(np.hypot(dx, dy))
    if denom < 1e-12:
        return float("inf")
    return abs(dy * point[0] - dx * point[1] + x2 * y1 - y2 * x1) / denom


# Draw detected lines with labels
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


# Choose equal length image segments
def choose_equal_length_segments(
    frame: np.ndarray,
    provided_segments: list[tuple[tuple[float, float], tuple[float, float]]] | None,
    draw_segments: bool,
    show_window: bool,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    if provided_segments is not None:
        return validate_equal_length_segments(provided_segments)

    if draw_segments or show_window:
        try:
            print(
                "Seleziona i segmenti uguali: clicca i 2 estremi del primo segmento "
                "e poi i 2 estremi del secondo.",
            )
            return draw_equal_length_segments_interactively(frame)
        except cv2.error:
            print("OpenCV GUI non disponibile: inserisci i segmenti dal terminale.")

    raw = input(
        "Inserisci due segmenti di uguale lunghezza reale come "
        "x1,y1,x2,y2; x1,y1,x2,y2: "
    )
    return validate_equal_length_segments(parse_equal_length_segments(raw))


# Parse selected equal length segments
def parse_equal_length_segments(
    raw: str,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    chunks = [chunk.strip() for chunk in raw.split(";") if chunk.strip()]
    if len(chunks) != 2:
        raise ValueError("Servono esattamente due segmenti separati da ';'.")

    segments = []
    for chunk in chunks:
        values = [float(value) for value in chunk.replace(",", " ").split()]
        if len(values) != 4:
            raise ValueError(f"Ogni segmento deve avere 4 coordinate, ricevuto: {chunk!r}.")
        segments.append(((values[0], values[1]), (values[2], values[3])))
    return segments


# Validate selected equal length segments
def validate_equal_length_segments(
    segments: list[tuple[tuple[float, float], tuple[float, float]]],
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    if len(segments) != 2:
        raise ValueError(f"Servono esattamente due segmenti, ricevuti {len(segments)}.")

    validated = []
    for segment in segments:
        if len(segment) != 2:
            raise ValueError(f"Segmento non valido: {segment}.")
        first = tuple(float(value) for value in segment[0])
        second = tuple(float(value) for value in segment[1])
        if len(first) != 2 or len(second) != 2:
            raise ValueError(f"Ogni endpoint deve avere due coordinate: {segment}.")
        if float(np.hypot(second[0] - first[0], second[1] - first[1])) <= 1e-6:
            raise ValueError("Gli estremi dei segmenti di uguale lunghezza devono essere distinti.")
        validated.append((first, second))

    return validated


# Collect equal length segments interactively
def draw_equal_length_segments_interactively(
    frame: np.ndarray,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    points: list[tuple[float, float]] = []
    window_name = "Seleziona 2 segmenti uguali: clicca 4 estremi"

    # Record selected image points
    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((float(x), float(y)))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    while len(points) < 4:
        preview = draw_equal_length_segments_preview(frame, points_to_segments(points))
        cv2.imshow(window_name, preview)
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):
            cv2.destroyWindow(window_name)
            raise ValueError("Selezione dei segmenti annullata.")

    cv2.destroyWindow(window_name)
    return validate_equal_length_segments(points_to_segments(points))


# Group selected points into segments
def points_to_segments(
    points: list[tuple[float, float]],
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    segments = []
    if len(points) >= 2:
        segments.append((points[0], points[1]))
    if len(points) >= 4:
        segments.append((points[2], points[3]))
    return segments


# Draw equal length segment previews
def draw_equal_length_segments_preview(
    frame: np.ndarray,
    segments: list[tuple[tuple[float, float], tuple[float, float]]],
) -> np.ndarray:
    preview = frame.copy()
    colors = [(0, 255, 255), (255, 0, 255)]
    for idx, segment in enumerate(segments):
        if len(segment) != 2:
            continue
        first = (int(round(segment[0][0])), int(round(segment[0][1])))
        second = (int(round(segment[1][0])), int(round(segment[1][1])))
        color = colors[idx % len(colors)]
        cv2.line(preview, first, second, color, 3, cv2.LINE_AA)
        cv2.circle(preview, first, 5, color, -1)
        cv2.circle(preview, second, 5, color, -1)
        midpoint = ((first[0] + second[0]) // 2, (first[1] + second[1]) // 2)
        cv2.putText(
            preview,
            f"L{idx + 1}",
            (midpoint[0] + 6, midpoint[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
            cv2.LINE_AA,
        )
    return preview


# Choose line groups for vanishing points
def choose_line_groups(
    lines: list[tuple[int, int, int, int]],
    provided_groups: list[list[int]] | None,
    legacy_ids: list[int] | None,
) -> list[list[int]]:
    # Group lines by vanishing direction
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
