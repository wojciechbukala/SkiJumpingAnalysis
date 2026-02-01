from dataclasses import dataclass
import numpy as np
import scipy.sparse
from scipy.sparse import csr_matrix,coo_matrix
from typing import Tuple,List
from scipy.sparse.linalg import lsqr


@dataclass
# w is a transformation matrix from frame_j to frame_i weighted by confidence weight
class Edge:
    frame_i_index: int
    frame_j_index: int
    warp_matrix: np.ndarray
    weight: float
@dataclass
class GraphMatrix:
    rows: list[int]
    cols: list[int]
    data: float
    def add(self, row: int, col: int, value: float) -> None:
        self.rows.append(row)
        self.cols.append(col)
        self.data.append(value)


# number of frames is total number of nodes
def create_graph(frames,edges: list[Edge],ref_frame: int =0,gauge_weight:float=1e3) -> Tuple[csr_matrix, np.ndarray]:
    n_equations = 6 * len(edges) + 6  # 6 equations per edge + 6 for ref frame
    n_unknowns= 6*len(frames)
    # use sparse matrix for efficiency
    A = GraphMatrix(rows=[], cols=[], data=[])
    

    b = np.zeros((n_equations, 1),dtype=float)  # 6N x 1 vector fill with 0
    row = 0
    for edge in edges:
        i = edge.frame_i_index
        j = edge.frame_j_index
        w = edge.warp_matrix
        weight = float(np.clip(edge.weight, 1e-12, 1.0))
        sqrt_weight = np.sqrt(weight)
        # set the variables for transformation matrix coefficients for easy access
        Acoef,Bcoef,Ccoef,Dcoef,Tx,Ty = w[0,0],w[0,1],w[0,2],w[1,0],w[1,1],w[1,2]
        #index to fill in A, it is 6 rows per frame
        iidx = 6 * i
        jidx = 6 * j
        # constrangt: H_j = H_i * W_ij
        #
        # H_i*W = [[a_i*A + b_i*C,  a_i*B + b_i*D,  a_i*Tx + b_i*Ty + tx_i],
        #          [c_i*A + d_i*C,  c_i*B + d_i*D,  c_i*Tx + d_i*Ty + ty_i]]
        #
        # = [[a_j, b_j, tx_j],
        #    [c_j, d_j, ty_j]]
        # Fill in the equations for frame i and j
        # a_i - A*a_j - C*d_j = 0
        A.add(row + 0, iidx + 0, sqrt_weight)
        A.add(row + 0, jidx + 0, -sqrt_weight * Acoef)
        A.add(row + 0, jidx + 1, -sqrt_weight * Ccoef)
        # b_i - B*a_j - D*d_j = 0
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
        # tx_i - tx_j - Tx*a_j - Tx*b_j =0    
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

        # gauge fix : X_ref=identity
    refFrameTrasf= np.eye(3,3,1)
    rr = 6*ref_frame
    for k in range(6):
        A.add(row,rr+k,gauge_weight)
        row+=1
    B= coo_matrix((A.data,(A.rows,A.cols)),shape=(n_equations,n_unknowns),dtype=np.float64).tocsr()
    return B,b

def solve(A: csr_matrix,b: np.ndarray) -> np.ndarray:
    res=lsqr(A,b.ravel(),atol=1e-12, btol=1e-12, iter_lim=10000)
    x=res[0]
    return x
    

def solve_graph(frames:int,edges:List[Edge],ref:int =0) ->List[np.ndarray]:
    A,bb=create_graph(frames,edges,ref)
    x=solve(A,bb)
    paramiters= x.reshape(frames,6)
    X: List[np.ndarray]=[]
    for i in range (frames):
        a,b,tx,c,d,ty=paramiters[i]
        Xi=np.ndarray([[a,b,tx],[c,d,ty],[0.,0.,1.]],dtype=np.float64)
        X.append(Xi)
    return X
