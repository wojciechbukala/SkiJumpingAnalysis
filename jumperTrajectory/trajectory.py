import numpy as np
import cv2
from transform.graphtr import Edge
from Mask import Masking

USE_GRAPH=True
def find_jumper_point(bbox: np.ndarray) -> tuple[int, int]:
    """
    Given a bounding box [x1, y1, x2, y2], return the center point (x, y).
    """
    x1, y1, x2, y2 = bbox
    x_center = int((x1 + x2) / 2)
    y_center = int((y2+y1)/2)
    return (x_center, y_center)


# def compute_trajectory(
#     transformations: list[EccResult],
#     ref_frame: int,
#     stop_frame: int,
# ) -> list[tuple[float, float,float]]:
#     """
#     Given a list of ECC results, compute the trajectory
#     of the jumper as a list of (x, y) points.
#     """
#     #C_t is the cumulative transformation matrix
#     # transformations[i].warp_matrix (from findTransformECC(template=prev, input=curr))
#     # maps coordinates from prev to curr.
#     # To express points of the current frame in the reference frame coordinates we
#     # must apply the inverse transforms cumulatively.

#     warp_by_pair: dict[tuple[int, int], np.ndarray] = {
#         (r.prev_idx, r.curr_idx): r.warp_matrix for r in transformations
#     }

#     # C maps current-frame coordinates -> ref-frame coordinates.
#     C = np.eye(3, dtype=np.float64)
#     trajectory: list[tuple[float,float,float]] = []

#     for frame in range(ref_frame, stop_frame):
#         # Update cumulative mapping for this frame (relative to ref_frame).
#         if frame > ref_frame:
#             W = warp_by_pair.get((frame - 1, frame))
#             if W is not None:
#                 try:
#                     C = C @ np.linalg.inv(W)
#                 except np.linalg.LinAlgError:
#                     # Degenerate transform; keep previous C.
#                     pass

#         bbox = Masking.get_bbox(frame)
#         if bbox is None:
#             continue

#         x, y = find_jumper_point(bbox)
#         pt = np.array([float(x), float(y), 1.0], dtype=np.float64)
#         mapped = C @ pt
#         # Ignore points that map to infinity.
#         if mapped[2] == 0:
#             continue

#         x_m = mapped[0] / mapped[2]
#         y_m = mapped[1] / mapped[2]
#         trajectory.append(( x_m, y_m, 1.0))

#     return trajectory

def compute_trajectory(
    transformations: list[np.ndarray],
    ref_frame: int,
    stop_frame: int,
    provider: Masking.JumperProvider,
) -> list[tuple[float, float,float]]:
    #C_t is the cumulative transformation matrix
    # transformations[i].warp_matrix (from findTransformECC(template=prev, input=curr))
    # maps coordinates from prev to curr.
    # To express points of the current frame in the reference frame coordinates we
    # must apply the inverse transforms cumulatively.

    warp_by_pair: dict[tuple[int, int], np.ndarray] = {
        (r.frame_i_index, r.frame_j_index): r.warp_matrix for r in transformations
    }

    # C maps current-frame coordinates -> ref-frame coordinates.
    C = np.eye(3, dtype=np.float64)
    trajectory: list[tuple[float,float,float]] = []

    for frame in range(ref_frame, stop_frame):
        # Update cumulative mapping for this frame (relative to ref_frame).
        if(USE_GRAPH==False):
            if frame > ref_frame:
                W = warp_by_pair.get((frame - 1, frame))
                if W is not None:
                    try:
                        C = C @ np.linalg.inv(W)
                    except np.linalg.LinAlgError:
                        # Degenerate transform; keep previous C.
                        pass
        else:
            C=transformations[frame]

        point = provider.get_point(frame)
        if point is None:
            continue

        x, y = point
        pt = np.array([float(x), float(y), 1.0], dtype=np.float64)
        mapped = C @ pt
        # Ignore points that map to infinity.
        if mapped[2] == 0:
            continue

        x_m = mapped[0] / mapped[2]
        y_m = mapped[1] / mapped[2]
        trajectory.append(( x_m, y_m, 1.0))

    return trajectory