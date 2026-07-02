# Image line extraction operations
# Step one converts the frame to grayscale and smooths it
# Step two applies contrast enhancement and Sobel magnitude before Canny edge detection
# Step three restricts detection to the lower search region controlled by LINE_SEARCH_TOP_RATIO
# Step four optionally runs LSD and Hough segment detectors
# Step five filters segments by length width and edge support
# Step six delegates final deduplication and ranking to tool_line select_lines
# Option USE_LSD_DETECTOR enables the line segment detector path
# Option USE_HOUGH_DETECTOR enables the probabilistic Hough path
# Option USE_VERTICAL_CLOSE_FOR_HOUGH adds vertical morphological closing before Hough
# Options LSD_MIN_LINE_LENGTH_PX LSD_MAX_LINE_WIDTH_PX and LSD_MIN_EDGE_SUPPORT_RATIO tune LSD filtering
# Options HOUGH_ATTEMPTS HOUGH_MIN_LINE_LENGTH_PX and HOUGH_MAX_LINE_GAP_PX tune Hough filtering
# Formula edge support ratio is hits divided by sampled points along the segment
# Formula segment length is sqrt((x2-x1)^2 + (y2-y1)^2)

from __future__ import annotations

import cv2
import numpy as np

from src.track_geometry.tool_line import select_lines


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


# Prepare image edges for detection
def preprocess_for_lines(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Enhance gradients for structural edge detection
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


# Detect and select image lines
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


# Compute the search region origin
def get_roi_y0(height: int) -> int:
    roi_y0 = int(height * LINE_SEARCH_TOP_RATIO)
    return max(0, min(height - 1, roi_y0))


# Detect segments with the line detector
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


# Detect segments with the Hough transform
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


# Prepare edges used for previews
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


# Measure edge evidence along a segment
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


# Round segment coordinates to pixels
def rounded_line(x1: float, y1: float, x2: float, y2: float) -> tuple[int, int, int, int]:
    return (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))
