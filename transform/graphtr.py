from dataclasses import dataclass
import numpy as np
from scipy.sparse import csr_matrix,coo_matrix
from scipy.sparse.linalg import lsqr
from typing import Any, Dict, List, Tuple


@dataclass
# warp_matrix maps frame_i (newer) -> frame_j (older) weighted by confidence
class Edge:
    frame_i_index: int
    frame_j_index: int
    warp_matrix: np.ndarray
    weight: float
@dataclass
class GraphMatrix:
    rows: list[int]
    cols: list[int]
    data: list[float]
    def add(self, row: int, col: int, value: float) -> None:
        self.rows.append(row)
        self.cols.append(col)
        self.data.append(value)


# 
def _mad(x: np.ndarray) -> float:
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)))


#
def _build_H_from_params(params: np.ndarray) -> List[np.ndarray]:
    H: List[np.ndarray] = []
    for a, b, c, d, tx, ty in params:
        H.append(np.array([[a, b, tx], [c, d, ty], [0.0, 0.0, 1.0]], dtype=np.float64))
    return H

def _edge_residual_vec(Hi: np.ndarray, Hj: np.ndarray, Wij: np.ndarray) -> np.ndarray:
    # Residual of: Hi ≈ Hj @ Wij (compare only the 2x3 affine block)
    pred = (Hj @ Wij)[:2, :]
    diff = Hi[:2, :] - pred
    return diff.reshape(-1)  # 6-vector



"""
    Build sparse system A x = b with constraints: H_i ≈ H_j W_ij and gauge fix H_ref = I.

    Notes:
    - We treat Edge.weight as a confidence already (often 0..1 from your RANSAC inlier ratio).
    - We penalize long edges (dt>1) so they help as loops but don't dominate.
    - We skip extremely weak long edges (better than adding almost-zero rows).
"""

# number of frames is total number of nodes
def create_graph(
    frames:int,
    edges: list[Edge],
    ref_frame: int =0,
    gauge_weight:float=1e3,
    *,
    long_alpha: float = 0.35,        # downweight long edges
    long_dt_power: float = 1.0,      # additionally divide by (dt^power)
    min_short_weight: float = 0.10,  # keep chain connectivity
    min_long_weight: float = 0.05,    # usually 0, long edges can disappear
    skip_long_below: float = 0.02,   # skip long edges that are too weak
) -> Tuple[csr_matrix, np.ndarray]:
    n_equations = 6 * len(edges) + 6  # 6 equations per edge + 6 for ref frame
    n_unknowns= 6*frames
    # use sparse matrix for efficiency
    A = GraphMatrix(rows=[], cols=[], data=[])
    

    b = np.zeros((n_equations, 1),dtype=float)  # 6N x 1 vector fill with 0
    row = 0
    max_inliers = max((float(e.weight) for e in edges), default=1.0)
    max_inliers = max(max_inliers, 1.0)

    for edge in edges:
        i = edge.frame_i_index
        j = edge.frame_j_index
        w = edge.warp_matrix
        dt = abs(i - j)
        base_w = float(max(0.0, edge.weight))

        if dt <= 1:
            # short edge: keep a minimum weight so graph stays connected
            w_eff = max(base_w, float(min_short_weight))
        else:
            # long edge: penalize
            w_eff = base_w * float(long_alpha) / float(max(1.0, dt ** float(long_dt_power)))
            w_eff = max(w_eff, float(min_long_weight))

            # skip very weak long edges (prevents near-zero rows -> numeric issues)
            #if w_eff < float(skip_long_below):
            #    continue

        sqrt_weight = float(np.sqrt(max(w_eff, 0.0)))

        # set the variables for transformation matrix coefficients for easy access
        Acoef,Bcoef,Ccoef,Dcoef,Tx,Ty = w[0,0],w[0,1],w[1,0],w[1,1],w[0,2],w[1,2]
        #index to fill in A, it is 6 rows per frame
        iidx = 6 * i
        jidx = 6 * j
        # constraint: H_i = H_j * W_ij
        #
        # H_j*W = [[a_j*A + b_j*C,  a_j*B + b_j*D,  a_j*Tx + b_j*Ty + tx_j],
        #          [c_j*A + d_j*C,  c_j*B + d_j*D,  c_j*Tx + d_j*Ty + ty_j]]
        #
        # = [[a_i, b_i, tx_i],
        #    [c_i, d_i, ty_i]]
        # Fill in the equations for frame i and j
        # a_i - A*a_j - C*b_j = 0
        A.add(row + 0, iidx + 0, sqrt_weight)
        A.add(row + 0, jidx + 0, -sqrt_weight * Acoef)
        A.add(row + 0, jidx + 1, -sqrt_weight * Ccoef)
        # b_i - B*a_j - D*b_j = 0
        A.add(row + 1, iidx + 1, sqrt_weight)
        A.add(row + 1, jidx + 0, -sqrt_weight * Bcoef)
        A.add(row + 1, jidx + 1, -sqrt_weight * Dcoef)
        # c_i - A*c_j - C*d_j =0
        A.add(row + 2, iidx + 2, sqrt_weight)
        A.add(row + 2, jidx + 2, -sqrt_weight * Acoef)
        A.add(row + 2, jidx + 3, -sqrt_weight * Ccoef)
        # d_i - B*c_j - D*d_j =0
        A.add(row + 3, iidx + 3, sqrt_weight)
        A.add(row + 3, jidx + 2, -sqrt_weight * Bcoef)
        A.add(row + 3, jidx + 3, -sqrt_weight * Dcoef)
        # tx_i - tx_j - Tx*a_j - Ty*b_j =0
        A.add(row+4,iidx + 4, sqrt_weight)
        A.add(row+4,jidx + 4, -sqrt_weight)
        A.add(row+4,jidx + 0, -sqrt_weight * Tx)
        A.add(row+4,jidx + 1, -sqrt_weight * Ty)
        # ty_i - ty_j - Tx*c_j - Ty*d_j =0
        A.add(row+5,iidx+5,sqrt_weight)
        A.add(row+5,jidx+2,-sqrt_weight*Tx)
        A.add(row+5,jidx+3,-sqrt_weight*Ty)
        A.add(row+5,jidx+5,-sqrt_weight)

        row += 6

    # gauge fix: H_ref = identity
    rr = 6*ref_frame
    target = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    for k, val in enumerate(target):
        A.add(row,rr+k,gauge_weight)
        b[row, 0] = gauge_weight * val
        row+=1
    B= coo_matrix((A.data,(A.rows,A.cols)),shape=(n_equations,n_unknowns),dtype=np.float64).tocsr()
    return B,b

def solve(A: csr_matrix,b: np.ndarray) -> np.ndarray:
    res=lsqr(A,b.ravel(),atol=1e-12, btol=1e-12, iter_lim=10000)
    x=res[0]
    return x
    




def _robust_reweight_edges(
    edges: List[Edge],
    H: List[np.ndarray],
    z_thresh: float = 3.5,
    tukey: bool = True,
    keep_short_floor: float = 0.15,  # keep short edges alive even if "outlier"
) -> Tuple[List[Edge], Dict[str, Any]]:
    """
    Reweight edges based on global consistency:
      residual r_e = Hi - Hj Wij  (2x3 block)
    Robust score via MAD; Tukey biweight inside threshold.
    """
    if not edges:
        return edges, {"n_outliers": 0, "n_edges": 0}

    R = np.zeros((len(edges), 6), dtype=np.float64)
    dt = np.zeros((len(edges),), dtype=np.int32)

    for k, e in enumerate(edges):
        i, j = int(e.frame_i_index), int(e.frame_j_index)
        dt[k] = abs(i - j)
        R[k, :] = _edge_residual_vec(H[i], H[j], e.warp_matrix)

    # normalize affine vs translation so tx/ty don't dominate by scale
    aff = R[:, :4]
    tr  = R[:, 4:6]
    s_aff = float(np.median(np.abs(aff))) + 1e-12
    s_tr  = float(np.median(np.abs(tr)))  + 1e-12

    N = np.sqrt(np.sum((aff / s_aff) ** 2, axis=1) + np.sum((tr / s_tr) ** 2, axis=1))

    med = float(np.median(N))
    mad = _mad(N) + 1e-12
    z = 0.6745 * (N - med) / mad
    az = np.abs(z)

    out = az > float(z_thresh)
    wrob = np.ones_like(az, dtype=np.float64)
    wrob[out] = 0.0

    if tukey:
        inl = ~out
        u = az[inl] / float(z_thresh)
        wrob[inl] = (1.0 - u * u) ** 2

    # keep short edges (dt==1) alive to preserve connectivity
    short = (dt <= 1)
    wrob[short] = np.maximum(wrob[short], float(keep_short_floor))

    new_edges: List[Edge] = []
    for e, wr in zip(edges, wrob):
        new_edges.append(Edge(
            frame_i_index=e.frame_i_index,
            frame_j_index=e.frame_j_index,
            warp_matrix=e.warp_matrix,
            weight=float(max(0.0, e.weight) * float(wr)),
        ))

    info = {
        "n_edges": int(len(edges)),
        "n_outliers": int(np.sum(out)),
        "median_norm": med,
        "mad_norm": mad,
        "mean_wrob": float(np.mean(wrob)),
        "min_wrob": float(np.min(wrob)),
    }
    return new_edges, info

    """
    Returns H_i (3x3) mapping frame i -> reference frame.

    robust=True enables a light IRLS:
      solve -> reweight edges by residual -> solve again (few iters).
    """
def solve_graph(
    frames: int,
    edges: List[Edge],
    ref: int = 0,
    index_offset: int = 0,  # kept for compatibility (unused)
    *,
    robust: bool = True,
    robust_iters: int = 3,
    z_thresh: float = 3.5,
    debug: bool = True,
) -> List[np.ndarray]:
    # pass-through knobs for create_graph:
    long_alpha: float = 0.35
    long_dt_power: float = 1.0
    min_short_weight: float = 0.05
    skip_long_below: float = 0.02
    if frames <= 0:
        return []
    if not edges:
        return [np.eye(3, dtype=np.float64) for _ in range(frames)]

    work_edges = list(edges)
    last_out = None
    iters = max(1, int(robust_iters)) if robust else 1

    H: List[np.ndarray] = []
    for it in range(iters):
        A, bb = create_graph(
            frames,
            work_edges,
            ref_frame=int(ref),
            gauge_weight=1e3,
            long_alpha=float(long_alpha),
            long_dt_power=float(long_dt_power),
            min_short_weight=float(min_short_weight),
            skip_long_below=float(skip_long_below),
        )
        x = solve(A, bb)
        params = x.reshape(frames, 6)
        H = _build_H_from_params(params)

        if not robust:
            return H

        work_edges, info = _robust_reweight_edges(
            work_edges,
            H,
            z_thresh=float(z_thresh),
            tukey=True,
            keep_short_floor=0.15,
        )

        if debug:
            print(
                f"[graphtr] IRLS it={it} "
                f"outliers={info['n_outliers']} mean_wrob={info['mean_wrob']:.3f}"
            )

        n_out = int(info.get("n_outliers", 0))
        if last_out is not None and n_out == last_out:
            break
        last_out = n_out

    return H
