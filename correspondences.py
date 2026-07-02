# Manual side-to-side correspondence discovery
# Step one loads the same frame used by calibration_phase
# Step two lets the user draw structural line features on the image
# Step three splits the features into left and right side groups with a user-drawn separator
# Step four builds automatic left-right match candidates and estimates their vanishing point with RANSAC
# Step five keeps a monotonic set of correspondences and saves CSV JSON and preview images
# Option SHOW_WINDOWS opens one preview for all features and one preview for accepted correspondences
# Option ASK_SIDE_SEPARATOR asks the user to draw a left-right separator polyline
# Formula image line from two points is [y1-y2 x2-x1 x1*y2-x2*y1]
# Formula vanishing point is the least-squares intersection of correspondence lines

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
import cv2
import numpy as np


from calibration_phase import FRAME_PATH as CALIBRATION_FRAME_PATH


FRAME_PATH = Path(CALIBRATION_FRAME_PATH)
OUTPUT_DIR = Path("outputGeometry")
FEATURES_CSV_PATH = OUTPUT_DIR / "correspondences_features.csv"
MATCHES_CSV_PATH = OUTPUT_DIR / "correspondences_matches.csv"
SUMMARY_JSON_PATH = OUTPUT_DIR / "correspondences_summary.json"
FEATURES_IMAGE_PATH = OUTPUT_DIR / "correspondences_features.jpg"
MATCHES_IMAGE_PATH = OUTPUT_DIR / "correspondences_matches.jpg"

SHOW_WINDOWS = True
ASK_SIDE_SEPARATOR = True
MANUAL_FEATURES_WINDOW_NAME = "correspondences - draw features"
FEATURES_WINDOW_NAME = "correspondences - selected features"
MATCHES_WINDOW_NAME = "correspondences - accepted matches"
SIDE_SEPARATOR_WINDOW_NAME = "correspondences - draw left/right separator"

MIN_FEATURE_LENGTH_PX = 30.0
MAX_FEATURES_TOTAL = 140
MAX_FEATURES_PER_SIDE = 70

MIN_SIDE_FEATURES = 4
MIN_MATCHES = 4
MIN_SIDE_SEPARATION_PX = 80.0
RANSAC_ORDER_TOLERANCE = 0.36
RANSAC_INLIER_THRESHOLD_PX = 18.0
RANSAC_ITERATIONS = 1800
RANSAC_SEED = 7
MATCH_MAX_RESIDUAL_PX = 32.0
MATCH_MAX_ORDER_DELTA = 0.38
MATCH_REWARD = 70.0
ORDER_COST_WEIGHT = 42.0
LENGTH_RATIO_COST_WEIGHT = 9.0
FORBID_VP_BETWEEN_SIDE_FEATURES = True
FORBID_VP_BETWEEN_MATCH_ENDPOINTS = True
VP_BETWEEN_SIDE_MARGIN_PX = 20.0
VP_BETWEEN_SIDE_Y_MARGIN_PX = 80.0
VP_BETWEEN_MATCH_ENDPOINT_MARGIN_PX = 16.0

LEFT_COLOR = (0, 210, 70)
RIGHT_COLOR = (0, 80, 255)
MATCH_COLOR = (190, 190, 190)
VP_COLOR = (255, 0, 255)
TEXT_COLOR = (255, 255, 255)
SEPARATOR_COLOR = (255, 255, 0)
SEPARATOR_POINT_COLOR = (0, 255, 255)
UNASSIGNED_FEATURE_COLOR = (180, 180, 180)


@dataclass(frozen=True)
class Feature:
    feature_id: int
    side: str
    x1: float
    y1: float
    x2: float
    y2: float
    mid_x: float
    mid_y: float
    length: float
    angle_deg: float
    vertical_deviation_deg: float
    edge_support: float
    mean_intensity: float
    score: float


@dataclass(frozen=True)
class MatchCandidate:
    left_index: int
    right_index: int
    left_rank: float
    right_rank: float
    line: np.ndarray
    line_length: float
    order_delta: float
    length_ratio: float


@dataclass(frozen=True)
class FeatureMatch:
    match_id: int
    left: Feature
    right: Feature
    residual_px: float
    line_length: float
    order_delta: float
    length_ratio: float


@dataclass(frozen=True)
class MatchResult:
    matches: list[FeatureMatch]
    vanishing_point: np.ndarray | None
    reversed_right_order: bool
    candidate_count: int
    inlier_count: int
    mean_residual_px: float | None
    median_residual_px: float | None
    max_residual_px: float | None
    message: str


def segment_angle_deg(x1: float, y1: float, x2: float, y2: float) -> float:
    return float(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0)


def vertical_deviation(angle_deg: float) -> float:
    return abs(angle_deg - 90.0)


def sample_segment_values(image: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    length = float(np.hypot(x2 - x1, y2 - y1))
    samples = max(8, int(round(length)))
    xs = np.clip(np.linspace(x1, x2, samples).round().astype(int), 0, image.shape[1] - 1)
    ys = np.clip(np.linspace(y1, y2, samples).round().astype(int), 0, image.shape[0] - 1)
    return image[ys, xs]


def feature_from_manual_segment(
    gray: np.ndarray,
    start: tuple[float, float],
    end: tuple[float, float],
) -> Feature | None:
    x1, y1 = start
    x2, y2 = end
    length = float(np.hypot(x2 - x1, y2 - y1))
    if length < MIN_FEATURE_LENGTH_PX:
        return None

    angle = segment_angle_deg(x1, y1, x2, y2)
    mean_intensity = float(np.mean(sample_segment_values(gray, x1, y1, x2, y2)))
    return Feature(
        feature_id=-1,
        side="unassigned",
        x1=float(x1),
        y1=float(y1),
        x2=float(x2),
        y2=float(y2),
        mid_x=0.5 * (x1 + x2),
        mid_y=0.5 * (y1 + y2),
        length=length,
        angle_deg=angle,
        vertical_deviation_deg=vertical_deviation(angle),
        edge_support=1.0,
        mean_intensity=mean_intensity,
        score=length,
    )


def rank_and_label_side_features(features: list[Feature], side: str) -> list[Feature]:
    best = sorted(features, key=lambda feature: feature.score, reverse=True)[:MAX_FEATURES_PER_SIDE]
    return [
        replace(feature, side=side, feature_id=idx)
        for idx, feature in enumerate(sorted(best, key=lambda item: (item.mid_y, item.mid_x)))
    ]


def split_features_by_side(features: list[Feature]) -> tuple[list[Feature], list[Feature]]:
    if len(features) < MIN_SIDE_FEATURES * 2:
        return [], []

    xs = np.asarray([feature.mid_x for feature in features], dtype=float)
    centers = np.percentile(xs, [30, 70]).astype(float)

    for _ in range(20):
        distances = np.abs(xs[:, None] - centers[None, :])
        labels = np.argmin(distances, axis=1)
        new_centers = centers.copy()
        for label in (0, 1):
            if np.any(labels == label):
                new_centers[label] = float(np.mean(xs[labels == label]))
        if np.allclose(new_centers, centers):
            break
        centers = new_centers

    left_label = int(np.argmin(centers))
    right_label = 1 - left_label
    distances = np.abs(xs[:, None] - centers[None, :])
    labels = np.argmin(distances, axis=1)

    left = [feature for feature, label in zip(features, labels) if label == left_label]
    right = [feature for feature, label in zip(features, labels) if label == right_label]

    return rank_and_label_side_features(left, "left"), rank_and_label_side_features(right, "right")


def separator_x_at_y(polyline: list[tuple[float, float]], y: float) -> float:
    if len(polyline) < 2:
        raise ValueError("The separator polyline needs at least two points.")

    intersections: list[float] = []
    for first, second in zip(polyline, polyline[1:]):
        x0, y0 = first
        x1, y1 = second
        if abs(y1 - y0) < 1e-9:
            if abs(y - y0) < 1e-9:
                intersections.extend([x0, x1])
            continue

        min_y = min(y0, y1)
        max_y = max(y0, y1)
        if min_y <= y <= max_y:
            t = (y - y0) / (y1 - y0)
            intersections.append(x0 + t * (x1 - x0))

    if intersections:
        return float(np.median(intersections))

    nearest = min(polyline, key=lambda point: abs(point[1] - y))
    return float(nearest[0])


def split_features_by_separator_polyline(
    features: list[Feature],
    separator_polyline: list[tuple[float, float]],
) -> tuple[list[Feature], list[Feature]]:
    left: list[Feature] = []
    right: list[Feature] = []

    for feature in features:
        boundary_x = separator_x_at_y(separator_polyline, feature.mid_y)
        if feature.mid_x < boundary_x:
            left.append(feature)
        else:
            right.append(feature)

    return rank_and_label_side_features(left, "left"), rank_and_label_side_features(right, "right")


def line_from_points(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    x1, y1 = [float(value) for value in first]
    x2, y2 = [float(value) for value in second]
    line = np.array([y1 - y2, x2 - x1, x1 * y2 - x2 * y1], dtype=float)
    norm = float(np.linalg.norm(line[:2]))
    if norm < 1e-12:
        raise ValueError("Degenerate image line.")
    return line / norm


def point_from_feature(feature: Feature) -> np.ndarray:
    return np.array([feature.mid_x, feature.mid_y], dtype=float)


def normalize_homogeneous_point(point: np.ndarray) -> np.ndarray | None:
    point = np.asarray(point, dtype=float).reshape(3)
    if not np.all(np.isfinite(point)):
        return None
    if abs(float(point[2])) < 1e-12:
        norm = float(np.linalg.norm(point[:2]))
        if norm < 1e-12:
            return None
        return point / norm
    return point / point[2]


def fit_vanishing_point(lines: list[np.ndarray]) -> np.ndarray | None:
    if len(lines) < 2:
        return None
    line_matrix = np.vstack(lines)
    _, _, vh = np.linalg.svd(line_matrix)
    return normalize_homogeneous_point(vh[-1, :])


def line_distance_to_vanishing_point(line: np.ndarray, vanishing_point: np.ndarray) -> float:
    if abs(float(vanishing_point[2])) < 1e-12:
        return abs(float(line @ vanishing_point))
    return abs(float(line @ vanishing_point)) / abs(float(vanishing_point[2]))


def is_vanishing_point_between_side_groups(
    vanishing_point: np.ndarray,
    left: list[Feature],
    right: list[Feature],
) -> bool:
    if not left or not right:
        return False

    point = normalize_homogeneous_point(vanishing_point)
    if point is None or abs(float(point[2])) < 1e-12:
        return False

    x = float(point[0])
    y = float(point[1])
    side_y = np.asarray([feature.mid_y for feature in [*left, *right]], dtype=float)
    y_min = float(np.min(side_y)) - VP_BETWEEN_SIDE_Y_MARGIN_PX
    y_max = float(np.max(side_y)) + VP_BETWEEN_SIDE_Y_MARGIN_PX
    if y < y_min or y > y_max:
        return False

    left_x = np.asarray([feature.mid_x for feature in left], dtype=float)
    right_x = np.asarray([feature.mid_x for feature in right], dtype=float)
    left_inner_x = float(np.median(left_x))
    right_inner_x = float(np.median(right_x))

    lower_x = min(left_inner_x, right_inner_x) + VP_BETWEEN_SIDE_MARGIN_PX
    upper_x = max(left_inner_x, right_inner_x) - VP_BETWEEN_SIDE_MARGIN_PX
    if lower_x >= upper_x:
        lower_x = min(left_inner_x, right_inner_x)
        upper_x = max(left_inner_x, right_inner_x)

    return lower_x <= x <= upper_x


def is_vanishing_point_between_feature_pair(
    vanishing_point: np.ndarray,
    left_feature: Feature,
    right_feature: Feature,
) -> bool:
    point = normalize_homogeneous_point(vanishing_point)
    if point is None or abs(float(point[2])) < 1e-12:
        return False

    vp_xy = point[:2].astype(float)
    left_xy = np.asarray([left_feature.mid_x, left_feature.mid_y], dtype=float)
    right_xy = np.asarray([right_feature.mid_x, right_feature.mid_y], dtype=float)
    segment = right_xy - left_xy
    segment_length_sq = float(segment @ segment)
    if segment_length_sq < 1e-12:
        return False

    segment_length = math.sqrt(segment_length_sq)
    t = float(((vp_xy - left_xy) @ segment) / segment_length_sq)
    margin_t = min(0.45, VP_BETWEEN_MATCH_ENDPOINT_MARGIN_PX / segment_length)
    return margin_t <= t <= 1.0 - margin_t


def is_vanishing_point_between_any_match(
    vanishing_point: np.ndarray,
    matches: list[FeatureMatch],
) -> bool:
    return any(
        is_vanishing_point_between_feature_pair(vanishing_point, match.left, match.right)
        for match in matches
    )


def is_plausible_vanishing_point(
    vanishing_point: np.ndarray,
    frame_shape: tuple[int, ...],
    left: list[Feature] | None = None,
    right: list[Feature] | None = None,
) -> bool:
    point = normalize_homogeneous_point(vanishing_point)
    if point is None or abs(float(point[2])) < 1e-12:
        return False
    height, width = frame_shape[:2]
    x = float(point[0])
    y = float(point[1])
    if not (-2.5 * width <= x <= 3.5 * width and -2.5 * height <= y <= 3.5 * height):
        return False
    if (
        FORBID_VP_BETWEEN_SIDE_FEATURES
        and left is not None
        and right is not None
        and is_vanishing_point_between_side_groups(point, left, right)
    ):
        return False
    return True


def build_match_candidates(
    left: list[Feature],
    right: list[Feature],
    reversed_right_order: bool,
) -> list[MatchCandidate]:
    right_order = list(reversed(right)) if reversed_right_order else right
    right_position = {feature.feature_id: idx for idx, feature in enumerate(right_order)}
    left_count = max(1, len(left) - 1)
    right_count = max(1, len(right_order) - 1)
    candidates: list[MatchCandidate] = []

    for left_idx, left_feature in enumerate(left):
        left_rank = left_idx / left_count
        left_point = point_from_feature(left_feature)

        for right_feature in right:
            right_idx = right_position[right_feature.feature_id]
            right_rank = right_idx / right_count
            order_delta = abs(left_rank - right_rank)
            if order_delta > RANSAC_ORDER_TOLERANCE:
                continue

            if right_feature.mid_x <= left_feature.mid_x + MIN_SIDE_SEPARATION_PX:
                continue

            right_point = point_from_feature(right_feature)
            line_length = float(np.linalg.norm(right_point - left_point))
            if line_length < MIN_SIDE_SEPARATION_PX:
                continue

            try:
                line = line_from_points(left_point, right_point)
            except ValueError:
                continue

            length_ratio = max(left_feature.length, right_feature.length) / max(
                1e-6,
                min(left_feature.length, right_feature.length),
            )
            candidates.append(
                MatchCandidate(
                    left_index=left_idx,
                    right_index=right_idx,
                    left_rank=left_rank,
                    right_rank=right_rank,
                    line=line,
                    line_length=line_length,
                    order_delta=order_delta,
                    length_ratio=length_ratio,
                ),
            )

    return candidates


def estimate_vanishing_point_ransac(
    candidates: list[MatchCandidate],
    frame_shape: tuple[int, ...],
    left: list[Feature],
    right: list[Feature],
    reversed_right_order: bool,
) -> tuple[np.ndarray | None, list[MatchCandidate]]:
    if len(candidates) < 2:
        return None, []

    rng = np.random.default_rng(RANSAC_SEED)
    right_order = list(reversed(right)) if reversed_right_order else right
    best_vp: np.ndarray | None = None
    best_inliers: list[MatchCandidate] = []
    best_median = float("inf")

    iterations = min(RANSAC_ITERATIONS, max(RANSAC_ITERATIONS // 4, len(candidates) * 20))
    for _ in range(iterations):
        first_idx, second_idx = rng.choice(len(candidates), size=2, replace=False)
        first_line = candidates[int(first_idx)].line
        second_line = candidates[int(second_idx)].line
        point = normalize_homogeneous_point(np.cross(first_line, second_line))
        if point is None or not is_plausible_vanishing_point(point, frame_shape, left, right):
            continue

        distances = np.asarray(
            [line_distance_to_vanishing_point(candidate.line, point) for candidate in candidates],
            dtype=float,
        )
        inlier_mask = []
        for candidate, distance in zip(candidates, distances):
            keep = bool(distance <= RANSAC_INLIER_THRESHOLD_PX)
            if keep and FORBID_VP_BETWEEN_MATCH_ENDPOINTS:
                left_feature = left[candidate.left_index]
                right_feature = right_order[candidate.right_index]
                keep = not is_vanishing_point_between_feature_pair(point, left_feature, right_feature)
            inlier_mask.append(keep)
        inliers = [candidate for candidate, keep in zip(candidates, inlier_mask) if keep]
        if len(inliers) < 2:
            continue

        inlier_array = np.asarray(inlier_mask, dtype=bool)
        median = float(np.median(distances[inlier_array]))
        if len(inliers) > len(best_inliers) or (len(inliers) == len(best_inliers) and median < best_median):
            best_vp = point
            best_inliers = inliers
            best_median = median

    if len(best_inliers) >= 2:
        refined = fit_vanishing_point([candidate.line for candidate in best_inliers])
        if refined is not None and is_plausible_vanishing_point(refined, frame_shape, left, right):
            best_vp = refined

    return best_vp, best_inliers


def candidate_cost(candidate: MatchCandidate, vanishing_point: np.ndarray) -> float:
    residual = line_distance_to_vanishing_point(candidate.line, vanishing_point)
    length_ratio_cost = abs(math.log(max(candidate.length_ratio, 1e-6)))
    return residual + ORDER_COST_WEIGHT * candidate.order_delta + LENGTH_RATIO_COST_WEIGHT * length_ratio_cost


def select_monotonic_matches(
    left: list[Feature],
    right: list[Feature],
    candidates: list[MatchCandidate],
    vanishing_point: np.ndarray,
    reversed_right_order: bool,
) -> list[FeatureMatch]:
    right_order = list(reversed(right)) if reversed_right_order else right
    costs: dict[tuple[int, int], tuple[float, MatchCandidate]] = {}

    for candidate in candidates:
        residual = line_distance_to_vanishing_point(candidate.line, vanishing_point)
        if residual > MATCH_MAX_RESIDUAL_PX or candidate.order_delta > MATCH_MAX_ORDER_DELTA:
            continue
        left_feature = left[candidate.left_index]
        right_feature = right_order[candidate.right_index]
        if (
            FORBID_VP_BETWEEN_MATCH_ENDPOINTS
            and is_vanishing_point_between_feature_pair(vanishing_point, left_feature, right_feature)
        ):
            continue
        cost = candidate_cost(candidate, vanishing_point)
        key = (candidate.left_index, candidate.right_index)
        existing = costs.get(key)
        if existing is None or cost < existing[0]:
            costs[key] = (cost, candidate)

    n = len(left)
    m = len(right_order)
    scores = np.zeros((n + 1, m + 1), dtype=float)
    choices = np.zeros((n + 1, m + 1), dtype=np.int8)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best_score = scores[i - 1, j]
            choice = 1
            if scores[i, j - 1] > best_score:
                best_score = scores[i, j - 1]
                choice = 2

            item = costs.get((i - 1, j - 1))
            if item is not None:
                cost, _ = item
                reward = MATCH_REWARD - cost
                if reward > 0.0 and scores[i - 1, j - 1] + reward > best_score:
                    best_score = scores[i - 1, j - 1] + reward
                    choice = 3

            scores[i, j] = best_score
            choices[i, j] = choice

    selected: list[MatchCandidate] = []
    i = n
    j = m
    while i > 0 and j > 0:
        choice = int(choices[i, j])
        if choice == 3:
            selected.append(costs[(i - 1, j - 1)][1])
            i -= 1
            j -= 1
        elif choice == 2:
            j -= 1
        else:
            i -= 1

    selected.reverse()

    matches = []
    for match_id, candidate in enumerate(selected):
        left_feature = left[candidate.left_index]
        right_feature = right_order[candidate.right_index]
        matches.append(
            FeatureMatch(
                match_id=match_id,
                left=left_feature,
                right=right_feature,
                residual_px=line_distance_to_vanishing_point(candidate.line, vanishing_point),
                line_length=candidate.line_length,
                order_delta=candidate.order_delta,
                length_ratio=candidate.length_ratio,
            ),
        )
    return matches


def compute_match_result_for_orientation(
    left: list[Feature],
    right: list[Feature],
    frame_shape: tuple[int, ...],
    reversed_right_order: bool,
) -> MatchResult:
    candidates = build_match_candidates(left, right, reversed_right_order)
    vp, inliers = estimate_vanishing_point_ransac(candidates, frame_shape, left, right, reversed_right_order)
    if vp is None:
        return MatchResult(
            matches=[],
            vanishing_point=None,
            reversed_right_order=reversed_right_order,
            candidate_count=len(candidates),
            inlier_count=0,
            mean_residual_px=None,
            median_residual_px=None,
            max_residual_px=None,
            message="Could not estimate a stable vanishing point from candidate matches.",
        )

    matches = select_monotonic_matches(left, right, inliers, vp, reversed_right_order)
    if len(matches) >= 2:
        lines = [line_from_points(point_from_feature(match.left), point_from_feature(match.right)) for match in matches]
        refined = fit_vanishing_point(lines)
        if (
            refined is not None
            and is_plausible_vanishing_point(refined, frame_shape, left, right)
            and not (
                FORBID_VP_BETWEEN_MATCH_ENDPOINTS
                and is_vanishing_point_between_any_match(refined, matches)
            )
        ):
            vp = refined
            matches = [
                replace(
                    match,
                    residual_px=line_distance_to_vanishing_point(
                        line_from_points(point_from_feature(match.left), point_from_feature(match.right)),
                        vp,
                    ),
                )
                for match in matches
            ]

    if FORBID_VP_BETWEEN_MATCH_ENDPOINTS and is_vanishing_point_between_any_match(vp, matches):
        return MatchResult(
            matches=[],
            vanishing_point=None,
            reversed_right_order=reversed_right_order,
            candidate_count=len(candidates),
            inlier_count=len(inliers),
            mean_residual_px=None,
            median_residual_px=None,
            max_residual_px=None,
            message="Rejected vanishing point because it falls between matched left/right features.",
        )

    residuals = np.asarray([match.residual_px for match in matches], dtype=float)
    return MatchResult(
        matches=matches,
        vanishing_point=vp,
        reversed_right_order=reversed_right_order,
        candidate_count=len(candidates),
        inlier_count=len(inliers),
        mean_residual_px=float(np.mean(residuals)) if len(residuals) else None,
        median_residual_px=float(np.median(residuals)) if len(residuals) else None,
        max_residual_px=float(np.max(residuals)) if len(residuals) else None,
        message="OK" if len(matches) >= MIN_MATCHES else f"Only {len(matches)} valid matches found.",
    )


def choose_best_match_result(left: list[Feature], right: list[Feature], frame_shape: tuple[int, ...]) -> MatchResult:
    results = [
        compute_match_result_for_orientation(left, right, frame_shape, reversed_right_order=False),
        compute_match_result_for_orientation(left, right, frame_shape, reversed_right_order=True),
    ]

    def sort_key(result: MatchResult) -> tuple[int, int, float]:
        median = result.median_residual_px if result.median_residual_px is not None else float("inf")
        return (len(result.matches), result.inlier_count, -median)

    results.sort(key=sort_key, reverse=True)
    return results[0]


def rounded_point(point: tuple[float, float] | np.ndarray) -> tuple[int, int]:
    return int(round(float(point[0]))), int(round(float(point[1])))


def draw_text_with_background(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int] = TEXT_COLOR,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.62
    thickness = 1
    size, baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    cv2.rectangle(image, (x - 5, y - size[1] - 6), (x + size[0] + 5, y + baseline + 5), (0, 0, 0), -1)
    cv2.putText(image, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def scaled_point(point: tuple[float, float], scale: float) -> tuple[int, int]:
    return int(round(point[0] * scale)), int(round(point[1] * scale))


def display_scale_for_frame(frame: np.ndarray) -> float:
    height, width = frame.shape[:2]
    return min(1.0, 1400.0 / float(width), 900.0 / float(height))


def draw_unassigned_features(image: np.ndarray, features: list[Feature], scale: float) -> None:
    for idx, feature in enumerate(features):
        p1 = scaled_point((feature.x1, feature.y1), scale)
        p2 = scaled_point((feature.x2, feature.y2), scale)
        midpoint = scaled_point((feature.mid_x, feature.mid_y), scale)
        cv2.line(image, p1, p2, UNASSIGNED_FEATURE_COLOR, 2, cv2.LINE_AA)
        cv2.circle(image, midpoint, 4, UNASSIGNED_FEATURE_COLOR, -1, cv2.LINE_AA)
        cv2.putText(
            image,
            str(idx),
            (midpoint[0] + 5, midpoint[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            UNASSIGNED_FEATURE_COLOR,
            1,
            cv2.LINE_AA,
        )


def draw_manual_features_editor_frame(
    frame: np.ndarray,
    features: list[Feature],
    pending_start: tuple[float, float] | None,
    mouse_position: tuple[float, float] | None,
    status_message: str,
    scale: float,
) -> np.ndarray:
    if scale != 1.0:
        display = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        display = frame.copy()

    draw_unassigned_features(display, features, scale)
    if pending_start is not None:
        start = scaled_point(pending_start, scale)
        cv2.circle(display, start, 6, SEPARATOR_POINT_COLOR, -1, cv2.LINE_AA)
        if mouse_position is not None:
            end = scaled_point(mouse_position, scale)
            cv2.line(display, start, end, SEPARATOR_COLOR, 2, cv2.LINE_AA)

    draw_text_with_background(display, "Disegna feature: click sinistro = primo/secondo punto", (24, 36))
    draw_text_with_background(
        display,
        "Enter/spazio/click destro conferma | u undo | c annulla punto | r reset | q/Esc esci",
        (24, 66),
    )
    draw_text_with_background(display, f"Feature selezionate: {len(features)}", (24, 96))
    if status_message:
        draw_text_with_background(display, status_message, (24, 126))
    return display


def request_manual_features(frame: np.ndarray) -> list[Feature]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scale = display_scale_for_frame(frame)
    state: dict[str, object] = {
        "features": [],
        "pending_start": None,
        "mouse_position": None,
        "done": False,
        "cancelled": False,
        "status": "",
    }

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        image_point = (float(x) / scale, float(y) / scale)
        if event == cv2.EVENT_MOUSEMOVE:
            state["mouse_position"] = image_point
            return

        features = state["features"]
        if not isinstance(features, list):
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            pending_start = state["pending_start"]
            if pending_start is None:
                state["pending_start"] = image_point
                state["status"] = "Secondo click per chiudere la feature."
                return

            feature = feature_from_manual_segment(gray, pending_start, image_point)
            state["pending_start"] = None
            if feature is None:
                state["status"] = f"Feature troppo corta: minimo {MIN_FEATURE_LENGTH_PX:.0f} px."
                return
            if len(features) >= MAX_FEATURES_TOTAL:
                state["status"] = f"Massimo {MAX_FEATURES_TOTAL} feature raggiunto."
                return
            features.append(feature)
            state["status"] = f"Feature aggiunta: {len(features)}."
        elif event == cv2.EVENT_RBUTTONDOWN:
            if state["pending_start"] is not None:
                state["pending_start"] = None
                state["status"] = "Punto iniziale annullato."
            elif features:
                state["done"] = True

    try:
        cv2.namedWindow(MANUAL_FEATURES_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(MANUAL_FEATURES_WINDOW_NAME, on_mouse)

        while True:
            features = state["features"]
            if not isinstance(features, list):
                features = []
            pending_start = state["pending_start"]
            mouse_position = state["mouse_position"]
            status = state["status"] if isinstance(state["status"], str) else ""
            preview = draw_manual_features_editor_frame(
                frame,
                features,
                pending_start if isinstance(pending_start, tuple) else None,
                mouse_position if isinstance(mouse_position, tuple) else None,
                status,
                scale,
            )
            cv2.imshow(MANUAL_FEATURES_WINDOW_NAME, preview)

            try:
                if cv2.getWindowProperty(MANUAL_FEATURES_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    state["cancelled"] = True
                    break
            except cv2.error:
                state["cancelled"] = True
                break

            key = cv2.waitKey(20) & 0xFF
            features = state["features"]
            if not isinstance(features, list):
                features = []

            if key in (13, 10, 32):
                if state["pending_start"] is None and features:
                    state["done"] = True
                else:
                    state["status"] = "Completa o annulla il segmento corrente prima di confermare."
            elif key == ord("u"):
                if state["pending_start"] is not None:
                    state["pending_start"] = None
                    state["status"] = "Punto iniziale annullato."
                elif features:
                    features.pop()
                    state["status"] = f"Ultima feature rimossa: {len(features)} rimaste."
            elif key == ord("c"):
                state["pending_start"] = None
                state["status"] = "Punto iniziale annullato."
            elif key == ord("r"):
                features.clear()
                state["pending_start"] = None
                state["status"] = "Feature resettate."
            elif key in (ord("q"), 27):
                state["cancelled"] = True

            if state["done"] or state["cancelled"]:
                break

        cv2.destroyWindow(MANUAL_FEATURES_WINDOW_NAME)
    except cv2.error as error:
        print(f"OpenCV GUI non disponibile per la selezione manuale delle feature: {error}")
        return []

    if state["cancelled"]:
        print("Selezione manuale feature annullata.")
        return []

    features = state["features"]
    if not isinstance(features, list):
        return []
    return features


def draw_separator_polyline(
    image: np.ndarray,
    separator_polyline: list[tuple[float, float]] | None,
    scale: float = 1.0,
) -> None:
    if not separator_polyline:
        return

    points = np.asarray([scaled_point(point, scale) for point in separator_polyline], dtype=np.int32)
    if len(points) >= 2:
        cv2.polylines(image, [points], False, SEPARATOR_COLOR, 3, cv2.LINE_AA)
    for point in points:
        cv2.circle(image, tuple(point), 5, SEPARATOR_POINT_COLOR, -1, cv2.LINE_AA)


def draw_separator_editor_frame(
    frame: np.ndarray,
    features: list[Feature],
    separator_polyline: list[tuple[float, float]],
    scale: float,
) -> np.ndarray:
    if scale != 1.0:
        display = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        display = frame.copy()

    draw_unassigned_features(display, features, scale)
    draw_separator_polyline(display, separator_polyline, scale)
    draw_text_with_background(display, "Disegna separatore sx/dx: click sinistro aggiunge punti", (24, 36))
    draw_text_with_background(
        display,
        "Enter/spazio/click destro conferma | u undo | r reset | q/Esc split automatico",
        (24, 66),
    )
    draw_text_with_background(display, f"Punti separatore: {len(separator_polyline)}", (24, 96))
    return display


def request_side_separator_polyline(frame: np.ndarray, features: list[Feature]) -> list[tuple[float, float]] | None:
    scale = display_scale_for_frame(frame)
    state: dict[str, object] = {"points": [], "done": False, "cancelled": False}

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        points = state["points"]
        if not isinstance(points, list):
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((float(x) / scale, float(y) / scale))
        elif event == cv2.EVENT_RBUTTONDOWN and len(points) >= 2:
            state["done"] = True

    try:
        cv2.namedWindow(SIDE_SEPARATOR_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(SIDE_SEPARATOR_WINDOW_NAME, on_mouse)

        while True:
            points = state["points"]
            if not isinstance(points, list):
                points = []
            preview = draw_separator_editor_frame(frame, features, points, scale)
            cv2.imshow(SIDE_SEPARATOR_WINDOW_NAME, preview)

            try:
                if cv2.getWindowProperty(SIDE_SEPARATOR_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    state["cancelled"] = True
                    break
            except cv2.error:
                state["cancelled"] = True
                break

            key = cv2.waitKey(20) & 0xFF
            if key in (13, 10, 32) and len(points) >= 2:
                state["done"] = True
            elif key == ord("u") and points:
                points.pop()
            elif key == ord("r"):
                points.clear()
            elif key in (ord("q"), 27):
                state["cancelled"] = True

            if state["done"] or state["cancelled"]:
                break

        cv2.destroyWindow(SIDE_SEPARATOR_WINDOW_NAME)
    except cv2.error as error:
        print(f"OpenCV GUI non disponibile per il separatore manuale: {error}")
        return None

    points = state["points"]
    if state["cancelled"] or not isinstance(points, list) or len(points) < 2:
        print("Separatore manuale non definito: uso split sx/dx automatico.")
        return None

    return [(float(x), float(y)) for x, y in points]


def draw_features_preview(
    frame: np.ndarray,
    left: list[Feature],
    right: list[Feature],
    separator_polyline: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    preview = frame.copy()
    draw_separator_polyline(preview, separator_polyline)
    for features, color, prefix in ((left, LEFT_COLOR, "L"), (right, RIGHT_COLOR, "R")):
        for feature in features:
            p1 = rounded_point((feature.x1, feature.y1))
            p2 = rounded_point((feature.x2, feature.y2))
            midpoint = rounded_point((feature.mid_x, feature.mid_y))
            cv2.line(preview, p1, p2, color, 2, cv2.LINE_AA)
            cv2.circle(preview, midpoint, 4, color, -1, cv2.LINE_AA)
            cv2.putText(
                preview,
                f"{prefix}{feature.feature_id}",
                (midpoint[0] + 5, midpoint[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                1,
                cv2.LINE_AA,
            )

    draw_text_with_background(preview, f"Detected features: left={len(left)} right={len(right)}", (24, 36))
    draw_text_with_background(preview, "Press any key for correspondences, q/Esc to close", (24, 66))
    return preview


def draw_matches_preview(
    frame: np.ndarray,
    matches: list[FeatureMatch],
    vanishing_point: np.ndarray | None,
    message: str,
    separator_polyline: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    preview = frame.copy()
    draw_separator_polyline(preview, separator_polyline)
    for match in matches:
        left_point = rounded_point((match.left.mid_x, match.left.mid_y))
        right_point = rounded_point((match.right.mid_x, match.right.mid_y))
        cv2.line(preview, left_point, right_point, MATCH_COLOR, 1, cv2.LINE_AA)
        cv2.circle(preview, left_point, 5, LEFT_COLOR, -1, cv2.LINE_AA)
        cv2.circle(preview, right_point, 5, RIGHT_COLOR, -1, cv2.LINE_AA)
        label_point = rounded_point(((match.left.mid_x + match.right.mid_x) * 0.5, (match.left.mid_y + match.right.mid_y) * 0.5))
        cv2.putText(
            preview,
            str(match.match_id),
            (label_point[0] + 4, label_point[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )

    if vanishing_point is not None and abs(float(vanishing_point[2])) > 1e-12:
        vp_xy = vanishing_point[:2] / vanishing_point[2]
        height, width = frame.shape[:2]
        if -width <= vp_xy[0] <= 2 * width and -height <= vp_xy[1] <= 2 * height:
            vp = rounded_point(vp_xy)
            cv2.drawMarker(preview, vp, VP_COLOR, cv2.MARKER_CROSS, 28, 2, cv2.LINE_AA)
            cv2.putText(preview, "VP", (vp[0] + 8, vp[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, VP_COLOR, 2, cv2.LINE_AA)

    draw_text_with_background(preview, f"Accepted correspondences: {len(matches)}", (24, 36))
    draw_text_with_background(preview, message, (24, 66))
    return preview


def write_features_csv(path: Path, features: list[Feature]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "feature_id",
                "side",
                "x1",
                "y1",
                "x2",
                "y2",
                "mid_x",
                "mid_y",
                "length",
                "angle_deg",
                "vertical_deviation_deg",
                "edge_support",
                "mean_intensity",
                "score",
            ],
        )
        for feature in features:
            writer.writerow(
                [
                    feature.feature_id,
                    feature.side,
                    feature.x1,
                    feature.y1,
                    feature.x2,
                    feature.y2,
                    feature.mid_x,
                    feature.mid_y,
                    feature.length,
                    feature.angle_deg,
                    feature.vertical_deviation_deg,
                    feature.edge_support,
                    feature.mean_intensity,
                    feature.score,
                ],
            )


def write_matches_csv(path: Path, matches: list[FeatureMatch]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "match_id",
                "left_feature_id",
                "right_feature_id",
                "left_mid_x",
                "left_mid_y",
                "right_mid_x",
                "right_mid_y",
                "residual_px",
                "line_length_px",
                "order_delta",
                "length_ratio",
            ],
        )
        for match in matches:
            writer.writerow(
                [
                    match.match_id,
                    match.left.feature_id,
                    match.right.feature_id,
                    match.left.mid_x,
                    match.left.mid_y,
                    match.right.mid_x,
                    match.right.mid_y,
                    match.residual_px,
                    match.line_length,
                    match.order_delta,
                    match.length_ratio,
                ],
            )


def write_summary_json(
    path: Path,
    frame: np.ndarray,
    left: list[Feature],
    right: list[Feature],
    result: MatchResult,
    success: bool,
    separator_polyline: list[tuple[float, float]] | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vanishing_point = None
    if result.vanishing_point is not None:
        vanishing_point = [float(value) for value in result.vanishing_point.tolist()]

    payload = {
        "status": "ok" if success else "failed",
        "message": result.message,
        "frame_path": str(FRAME_PATH),
        "image_width": int(frame.shape[1]),
        "image_height": int(frame.shape[0]),
        "feature_source": "manual",
        "left_feature_count": len(left),
        "right_feature_count": len(right),
        "candidate_match_count": result.candidate_count,
        "ransac_inlier_count": result.inlier_count,
        "accepted_match_count": len(result.matches),
        "reversed_right_order": result.reversed_right_order,
        "used_manual_side_separator": separator_polyline is not None,
        "side_separator_polyline": [
            [float(x), float(y)] for x, y in separator_polyline
        ]
        if separator_polyline is not None
        else None,
        "vanishing_point": vanishing_point,
        "mean_residual_px": result.mean_residual_px,
        "median_residual_px": result.median_residual_px,
        "max_residual_px": result.max_residual_px,
        "outputs": {
            "features_csv": str(FEATURES_CSV_PATH),
            "matches_csv": str(MATCHES_CSV_PATH),
            "features_image": str(FEATURES_IMAGE_PATH),
            "matches_image": str(MATCHES_IMAGE_PATH),
        },
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def show_preview_windows(features_preview: np.ndarray, matches_preview: np.ndarray) -> None:
    try:
        cv2.namedWindow(FEATURES_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.imshow(FEATURES_WINDOW_NAME, features_preview)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow(FEATURES_WINDOW_NAME)
        if key in (ord("q"), 27):
            cv2.destroyAllWindows()
            return

        cv2.namedWindow(MATCHES_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.imshow(MATCHES_WINDOW_NAME, matches_preview)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error as error:
        print(f"OpenCV GUI non disponibile: {error}")


def empty_result(message: str) -> MatchResult:
    return MatchResult(
        matches=[],
        vanishing_point=None,
        reversed_right_order=False,
        candidate_count=0,
        inlier_count=0,
        mean_residual_px=None,
        median_residual_px=None,
        max_residual_px=None,
        message=message,
    )


def run(show_windows: bool = SHOW_WINDOWS, ask_side_separator: bool = ASK_SIDE_SEPARATOR) -> int:
    frame = cv2.imread(str(FRAME_PATH))
    if frame is None:
        raise FileNotFoundError(f"Frame non trovato o non leggibile: {FRAME_PATH}")

    raw_features = request_manual_features(frame)
    separator_polyline = None
    if raw_features and ask_side_separator:
        separator_polyline = request_side_separator_polyline(frame, raw_features)

    if not raw_features:
        left, right = [], []
        result = empty_result("No manual features selected.")
    elif separator_polyline is not None:
        left, right = split_features_by_separator_polyline(raw_features, separator_polyline)
    else:
        left, right = split_features_by_side(raw_features)
    features = [*left, *right]

    if not raw_features:
        pass
    elif len(left) < MIN_SIDE_FEATURES or len(right) < MIN_SIDE_FEATURES:
        result = empty_result(
            f"Not enough side features: left={len(left)} right={len(right)} minimum={MIN_SIDE_FEATURES}.",
        )
    else:
        result = choose_best_match_result(left, right, frame.shape)

    success = len(result.matches) >= MIN_MATCHES and result.vanishing_point is not None
    if not success and result.message == "OK":
        result = replace(result, message=f"Only {len(result.matches)} valid matches found; minimum is {MIN_MATCHES}.")

    features_preview = draw_features_preview(frame, left, right, separator_polyline)
    matches_preview = draw_matches_preview(frame, result.matches, result.vanishing_point, result.message, separator_polyline)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(FEATURES_IMAGE_PATH), features_preview)
    cv2.imwrite(str(MATCHES_IMAGE_PATH), matches_preview)
    write_features_csv(FEATURES_CSV_PATH, features)
    write_matches_csv(MATCHES_CSV_PATH, result.matches)
    write_summary_json(SUMMARY_JSON_PATH, frame, left, right, result, success, separator_polyline)

    print(f"Frame usato: {FRAME_PATH}")
    print(f"Feature selezionate: left={len(left)} right={len(right)}")
    print(f"Separatore manuale sx/dx: {'si' if separator_polyline is not None else 'no'}")
    print(f"Corrispondenze accettate: {len(result.matches)}")
    if result.vanishing_point is not None:
        print("Punto di fuga stimato:", result.vanishing_point)
    print(f"Preview feature: {FEATURES_IMAGE_PATH}")
    print(f"Preview corrispondenze: {MATCHES_IMAGE_PATH}")
    print(f"Summary: {SUMMARY_JSON_PATH}")

    if show_windows:
        show_preview_windows(features_preview, matches_preview)

    if not success:
        print(f"Correspondences non riuscita: {result.message}")
        return 1
    return 0


def main() -> int:
    return run(SHOW_WINDOWS)


if __name__ == "__main__":
    raise SystemExit(main())
