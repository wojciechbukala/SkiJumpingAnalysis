"""
Image alignment using Enhanced Correlation Coefficient (ECC) Maximization

Original url:
https://www.learnopencv.com/image-alignment-ecc-in-opencv-c-python/

Motion models:
    - Translation
    - Euclidean
    - Affine
    - Homography

cv2.findTransformECC():
    - Read the images.
    - Convert them to grayscale.
    - Pick a motion model you want to estimate.
    - Allocate space (warp_matrix) to store the motion model.
    - Define a termination criteria that tells the algorithm when to stop.
    - Estimate the warp matrix using findTransformECC.
    - Apply the warp matrix to one of the images to align it with the other image.

"""

# Import the packages
from collections.abc import Iterable
from dataclasses import dataclass
from typing import List, Optional, Tuple
from Mask import Masking
import cv2
import numpy as np

DEFAULT_SAMPLE_RATE = 1

@dataclass
class EccResult:
    prev_idx: int
    curr_idx: int
    warp_matrix: np.ndarray          # 2x3 (AFFINE) or 3x3 (HOMOGRAPHY)
    correlation_coefficient: float

# Yield frames from the video capture at a specified sample rate.
# frames are converted to grayscale for ECC processing.
def iter_frames(cap, sample_rate: int = DEFAULT_SAMPLE_RATE) -> Iterable[Tuple[int, "cv2.Mat"]]:
    if sample_rate < 1:
        raise ValueError("sample_rate must be >= 1.")

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if sample_rate == 1 or frame_idx % sample_rate == 0:
            #convert frame to gray scale
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            yield frame_idx, frame
        frame_idx += 1

# evaluate ECC between two frames, return the transformation that aligns im2 to im1
def ecc_2frame(
    prev_idx: int,
    im1: np.ndarray,
    im2: np.ndarray,
    warp_mode: int = cv2.MOTION_AFFINE,
    number_of_iterations: int = 200, # maximum number of iterations to find the warp matrix
    termination_eps: float = 1e-6, # treshold of convergence
    gaussFiltSize: int = 5,
) -> Tuple[float, np.ndarray]:
    # Convert images to float32 with normalized to [0,1]
    im1 = im1.astype(np.float32)/255.0
    im2 = im2.astype(np.float32)/255.0
    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,number_of_iterations,  termination_eps)
    mask1= Masking.create_mask(im1.shape,prev_idx)
    mask2= Masking.create_mask(im2.shape,prev_idx+1)
    # Combine masks with bitwise AND
    mask = cv2.bitwise_and(mask1, mask2)
    # Run the ECC algorithm. The results are stored in warp_matrix.
    # cc is the correlation coefficient
    (cc, warp_matrix) = cv2.findTransformECC(im1, im2, warp_matrix, warp_mode, criteria, inputMask=mask, gaussFiltSize=gaussFiltSize)

    return cc, warp_matrix


# Detect ECC motion over a sequence of frames from the video capture.
# Returns a list of EccResult objects.
def detect_ecc_motion_sequence(
    cap,
) -> list[EccResult]:
    generatorFrame = iter_frames(cap)

    try:
        prev_idx, prev = next(generatorFrame)
    except StopIteration:
        return []
    results: List[EccResult] = []
    for curr_idx, curr in generatorFrame:
        try:
            cc,warp_matrix = ecc_2frame(
                prev_idx,
                prev,
                curr,
            )
            results.append(EccResult(
                prev_idx=prev_idx,
                curr_idx=curr_idx,
                warp_matrix=warp_matrix,
                correlation_coefficient=cc,
            ))
        except cv2.error as e:
            print(f"ECC alignment failed between frames {prev_idx} and {curr_idx}: {e}")
            # Skip this pair and continue
            pass
        prev_idx, prev = curr_idx, curr
    return results



