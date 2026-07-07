# Camera calibration operations
# Step one accepts selected orthogonal vanishing point pairs
# Step two estimates the intrinsic matrix K with zero skew and independent focal scales fx and fy
# Step three optionally adds an equal-length metric constraint to estimate the principal point
# Step four validates focal scales and principal point against the image size
# Option principal_point fixes cx and cy when known
# Formula for known principal point is ((u1-cx)(u2-cx))/fx^2 + ((v1-cy)(v2-cy))/fy^2 + 1 = 0
# Formula uses ax = 1/fx^2 and ay = 1/fy^2 so each pair gives ax dx1 dx2 + ay dy1 dy2 = -1

from __future__ import annotations

import math

import numpy as np


class CalibrationError(RuntimeError):
    pass


# Estimate intrinsics from orthogonal vanishing pairs
def estimate_k_from_orthogonal_vanishing_pairs(
    orthogonal_pairs,
    image_shape: tuple[int, int] | None = None,
    principal_point: tuple[float, float] | None = None,
) -> np.ndarray:
    pairs = _as_vanishing_pairs(orthogonal_pairs)
    cx, cy = _resolve_principal_point(principal_point, image_shape)
    return _estimate_zero_skew_k_known_principal_point(pairs, cx, cy, image_shape)


# Estimate intrinsics from orthogonal pairs and a metric equal-length constraint
def estimate_k_from_orthogonal_vanishing_pairs_and_equal_lengths(
    orthogonal_pairs,
    equal_length_segments,
    plane_normal_vanishing_point,
    image_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    pairs = _as_vanishing_pairs(orthogonal_pairs)
    segments = _as_equal_length_segments(equal_length_segments)
    normal_vp = _as_inhomogeneous_point(plane_normal_vanishing_point)

    if len(pairs) < 3:
        raise CalibrationError(
            f"At least three orthogonal vanishing-point pairs are required, got {len(pairs)}.",
        )

    try:
        from scipy.optimize import least_squares
    except ImportError:
        least_squares = None

    height, width = image_shape if image_shape is not None else _infer_image_shape_from_points(pairs, segments, normal_vp)
    max_dim = float(max(width, height))
    center = (width * 0.5, height * 0.5)

    starts = []
    try:
        K_center = _estimate_zero_skew_k_known_principal_point(pairs, center[0], center[1], image_shape)
        starts.append((float(K_center[0, 0]), float(K_center[1, 1]), center[0], center[1]))
    except CalibrationError:
        pass
    starts.extend(
        [
            (max_dim, max_dim, center[0], center[1]),
            (1.5 * max_dim, 1.5 * max_dim, center[0], center[1]),
            (2.0 * max_dim, 1.5 * max_dim, center[0], center[1]),
            (1.5 * max_dim, 2.0 * max_dim, center[0], center[1]),
        ],
    )

    lower = [
        math.log(0.15 * max_dim),
        math.log(0.15 * max_dim),
        -2.0 * width / max_dim,
        -2.0 * height / max_dim,
    ]
    upper = [
        math.log(12.0 * max_dim),
        math.log(12.0 * max_dim),
        3.0 * width / max_dim,
        3.0 * height / max_dim,
    ]

    best_result = None
    for fx, fy, cx, cy in starts:
        x0 = np.array([math.log(fx), math.log(fy), cx / max_dim, cy / max_dim], dtype=float)
        if least_squares is not None:
            result = least_squares(
                _equal_length_calibration_residuals,
                x0,
                bounds=(lower, upper),
                args=(pairs, segments, normal_vp, max_dim),
                xtol=1e-12,
                ftol=1e-12,
                gtol=1e-12,
                max_nfev=4000,
            )
        else:
            result = _least_squares_numpy(
                _equal_length_calibration_residuals,
                x0,
                np.asarray(lower, dtype=float),
                np.asarray(upper, dtype=float),
                args=(pairs, segments, normal_vp, max_dim),
            )
        if best_result is None or float(result.cost) < float(best_result.cost):
            best_result = result

    if best_result is None or not best_result.success:
        raise CalibrationError("Equal-length calibration did not converge.")

    fx, fy, cx, cy = _decode_equal_length_parameters(best_result.x, max_dim)
    residual_norm = float(np.linalg.norm(_equal_length_calibration_residuals(best_result.x, pairs, segments, normal_vp, max_dim)))
    if residual_norm > 5e-2:
        raise CalibrationError(f"Equal-length calibration residual too high: {residual_norm:.3e}.")

    _validate_k_geometry(fx, fy, cx, cy, (height, width))
    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


class _LeastSquaresResult:
    def __init__(self, x: np.ndarray, cost: float, success: bool) -> None:
        self.x = x
        self.cost = cost
        self.success = success


def _least_squares_numpy(
    residual_fn,
    x0: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    args: tuple,
) -> _LeastSquaresResult:
    x = np.clip(np.asarray(x0, dtype=float), lower, upper)
    residual = np.asarray(residual_fn(x, *args), dtype=float)
    best_cost = 0.5 * float(residual @ residual)
    damping = 1e-3

    for _ in range(400):
        jacobian = _finite_difference_jacobian(residual_fn, x, lower, upper, args)
        system = jacobian.T @ jacobian + damping * np.eye(len(x), dtype=float)
        rhs = -(jacobian.T @ residual)
        try:
            step = np.linalg.solve(system, rhs)
        except np.linalg.LinAlgError:
            step, _, _, _ = np.linalg.lstsq(system, rhs, rcond=None)

        if float(np.linalg.norm(step)) < 1e-12:
            return _LeastSquaresResult(x, best_cost, True)

        improved = False
        for scale in (1.0, 0.5, 0.25, 0.1, 0.05, 0.01):
            candidate = np.clip(x + scale * step, lower, upper)
            candidate_residual = np.asarray(residual_fn(candidate, *args), dtype=float)
            candidate_cost = 0.5 * float(candidate_residual @ candidate_residual)
            if np.isfinite(candidate_cost) and candidate_cost < best_cost:
                x = candidate
                residual = candidate_residual
                best_cost = candidate_cost
                damping = max(damping * 0.5, 1e-9)
                improved = True
                break

        if not improved:
            damping = min(damping * 10.0, 1e9)

        if best_cost < 1e-24:
            return _LeastSquaresResult(x, best_cost, True)

    return _LeastSquaresResult(x, best_cost, best_cost < 1e-16)


def _finite_difference_jacobian(
    residual_fn,
    x: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    args: tuple,
) -> np.ndarray:
    base = np.asarray(residual_fn(x, *args), dtype=float)
    jacobian = np.zeros((len(base), len(x)), dtype=float)
    for idx in range(len(x)):
        step = 1e-6 * max(1.0, abs(float(x[idx])))
        forward = x.copy()
        backward = x.copy()
        forward[idx] = min(float(upper[idx]), float(x[idx] + step))
        backward[idx] = max(float(lower[idx]), float(x[idx] - step))
        denom = float(forward[idx] - backward[idx])
        if abs(denom) < 1e-15:
            continue
        residual_forward = np.asarray(residual_fn(forward, *args), dtype=float)
        residual_backward = np.asarray(residual_fn(backward, *args), dtype=float)
        jacobian[:, idx] = (residual_forward - residual_backward) / denom
    return jacobian


# Estimate scales with a known principal point
def _estimate_zero_skew_k_known_principal_point(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    cx: float,
    cy: float,
    image_shape: tuple[int, int] | None,
) -> np.ndarray:
    # With zero skew and independent horizontal and vertical focal scales:
    # ((u1-cx)(u2-cx))/fx^2 + ((v1-cy)(v2-cy))/fy^2 + 1 = 0.
    # Let ax = 1/fx^2 and ay = 1/fy^2, then each orthogonal pair is linear
    # in ax and ay once the principal point is fixed.
    if len(pairs) < 2:
        raise CalibrationError(
            f"At least two orthogonal vanishing-point pairs are required, got {len(pairs)}.",
        )

    matrix = []
    rhs = []
    for v1, v2 in pairs:
        matrix.append([(v1[0] - cx) * (v2[0] - cx), (v1[1] - cy) * (v2[1] - cy)])
        rhs.append(-1.0)

    A = np.asarray(matrix, dtype=float)
    b = np.asarray(rhs, dtype=float)
    if np.linalg.matrix_rank(A) < 2:
        raise CalibrationError("Orthogonal vanishing-point pairs are degenerate for fx/fy estimation.")

    solution, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    inv_fx_squared, inv_fy_squared = [float(value) for value in solution]

    if not np.isfinite(inv_fx_squared) or inv_fx_squared <= 0.0:
        raise CalibrationError("Orthogonal pairs produced a non-positive fx scale.")
    if not np.isfinite(inv_fy_squared) or inv_fy_squared <= 0.0:
        raise CalibrationError("Orthogonal pairs produced a non-positive fy scale.")

    fx = math.sqrt(1.0 / inv_fx_squared)
    fy = math.sqrt(1.0 / inv_fy_squared)
    if image_shape is not None:
        _validate_k_geometry(fx, fy, cx, cy, image_shape)

    if len(pairs) > 2:
        mean_residual = float(np.sqrt(np.mean((A @ solution - b) ** 2)))
        if mean_residual > 0.5:
            raise CalibrationError(
                f"Orthogonal-pair equations are inconsistent, mean residual={mean_residual:.3f}.",
            )

    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


# Convert input pairs to normalized arrays
def _as_vanishing_pairs(orthogonal_pairs) -> list[tuple[np.ndarray, np.ndarray]]:
    pairs = []
    for raw_pair in orthogonal_pairs:
        if len(raw_pair) != 2:
            raise CalibrationError("Each orthogonal pair must contain exactly two vanishing points.")
        pairs.append((_as_inhomogeneous_point(raw_pair[0]), _as_inhomogeneous_point(raw_pair[1])))
    return pairs


def _as_equal_length_segments(equal_length_segments) -> list[tuple[np.ndarray, np.ndarray]]:
    if len(equal_length_segments) != 2:
        raise CalibrationError("Exactly two equal-length segments are required.")

    segments = []
    for raw_segment in equal_length_segments:
        if len(raw_segment) != 2:
            raise CalibrationError("Each equal-length segment must contain two endpoints.")
        first = np.asarray(raw_segment[0], dtype=float).reshape(-1)
        second = np.asarray(raw_segment[1], dtype=float).reshape(-1)
        if first.size != 2 or second.size != 2:
            raise CalibrationError("Equal-length segment endpoints must have two image coordinates.")
        if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
            raise CalibrationError("Equal-length segment contains non-finite coordinates.")
        if float(np.linalg.norm(second - first)) <= 1e-9:
            raise CalibrationError("Equal-length segment endpoints must be distinct.")
        segments.append((first.astype(float), second.astype(float)))
    return segments


def _infer_image_shape_from_points(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    segments: list[tuple[np.ndarray, np.ndarray]],
    normal_vp: np.ndarray,
) -> tuple[int, int]:
    points = [normal_vp]
    for first, second in pairs:
        points.extend([first, second])
    for first, second in segments:
        points.extend([first, second])
    stacked = np.vstack(points)
    width = int(max(1.0, float(np.nanmax(stacked[:, 0])) * 2.0))
    height = int(max(1.0, float(np.nanmax(stacked[:, 1])) * 2.0))
    return height, width


def _decode_equal_length_parameters(parameters: np.ndarray, max_dim: float) -> tuple[float, float, float, float]:
    fx = math.exp(float(parameters[0]))
    fy = math.exp(float(parameters[1]))
    cx = float(parameters[2]) * max_dim
    cy = float(parameters[3]) * max_dim
    return fx, fy, cx, cy


def _equal_length_calibration_residuals(
    parameters: np.ndarray,
    pairs: list[tuple[np.ndarray, np.ndarray]],
    segments: list[tuple[np.ndarray, np.ndarray]],
    normal_vp: np.ndarray,
    max_dim: float,
) -> np.ndarray:
    fx, fy, cx, cy = _decode_equal_length_parameters(parameters, max_dim)

    residuals = []
    for v1, v2 in pairs:
        residuals.append(((v1[0] - cx) * (v2[0] - cx)) / (fx * fx) + ((v1[1] - cy) * (v2[1] - cy)) / (fy * fy) + 1.0)

    try:
        length_squares = [
            _plane_segment_length_square(segment, normal_vp, fx, fy, cx, cy)
            for segment in segments
        ]
    except CalibrationError:
        residuals.append(1e6)
        return np.asarray(residuals, dtype=float)

    scale = max(length_squares[0], length_squares[1], 1e-12)
    residuals.append((length_squares[0] - length_squares[1]) / scale)
    return np.asarray(residuals, dtype=float)


def _plane_segment_length_square(
    segment: tuple[np.ndarray, np.ndarray],
    normal_vp: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> float:
    first, second = segment
    K_inv = np.array(
        [
            [1.0 / fx, 0.0, -cx / fx],
            [0.0, 1.0 / fy, -cy / fy],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    normal = K_inv @ np.array([normal_vp[0], normal_vp[1], 1.0], dtype=float)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm < 1e-12:
        raise CalibrationError("Degenerate plane normal direction.")
    normal = normal / normal_norm

    def plane_point(image_point: np.ndarray) -> np.ndarray:
        ray = K_inv @ np.array([image_point[0], image_point[1], 1.0], dtype=float)
        denom = float(np.dot(normal, ray))
        if abs(denom) < 1e-12:
            raise CalibrationError("Image ray is parallel to the equal-length plane.")
        return ray / denom

    delta = plane_point(second) - plane_point(first)
    length_square = float(np.dot(delta, delta))
    if not np.isfinite(length_square):
        raise CalibrationError("Invalid lifted segment length.")
    return length_square


# Convert one point to image coordinates
def _as_inhomogeneous_point(point) -> np.ndarray:
    arr = np.asarray(point, dtype=float).reshape(-1)
    if arr.size == 2:
        result = arr
    elif arr.size == 3:
        if abs(float(arr[2])) < 1e-12:
            raise CalibrationError("Vanishing point at infinity cannot estimate finite K.")
        result = arr[:2] / arr[2]
    else:
        raise CalibrationError("Vanishing points must have 2 or 3 coordinates.")

    if not np.all(np.isfinite(result)):
        raise CalibrationError("Vanishing point contains non-finite values.")
    return result.astype(float)


# Resolve the principal point coordinates
def _resolve_principal_point(
    principal_point: tuple[float, float] | None,
    image_shape: tuple[int, int] | None,
) -> tuple[float, float]:
    if principal_point is not None:
        cx, cy = [float(value) for value in principal_point]
        return cx, cy

    if image_shape is None:
        raise CalibrationError(
            "principal_point or image_shape is required to resolve the principal point.",
        )

    height, width = image_shape
    return width * 0.5, height * 0.5


# Validate estimated intrinsic geometry
def _validate_k_geometry(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    image_shape: tuple[int, int],
) -> None:
    height, width = image_shape
    max_dim = float(max(width, height))

    if fx < 0.15 * max_dim or fx > 12.0 * max_dim:
        raise CalibrationError(f"Estimated focal length is implausible: fx={fx:.3f}.")
    if fy < 0.15 * max_dim or fy > 12.0 * max_dim:
        raise CalibrationError(f"Estimated focal length is implausible: fy={fy:.3f}.")

    if cx < -2.0 * width or cx > 3.0 * width or cy < -2.0 * height or cy > 3.0 * height:
        raise CalibrationError(
            f"Estimated principal point is implausible: cx={cx:.3f}, cy={cy:.3f}.",
        )
