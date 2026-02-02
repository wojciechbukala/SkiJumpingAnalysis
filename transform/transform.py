# Import the packages
from collections.abc import Iterable
from typing import List,Tuple
from Mask import Masking
import cv2
import numpy as np
from transform.graphtr import Edge
from collections import deque
LOOKBACK = 0  # "long edge" every time we have lookback+1 frames in memory
DEFAULT_SAMPLE_RATE = 1
# Global switch: "ecc" use ECC for all frames
# anything else use SIFT + RANSAC for all frames
GLOBAL_VAR = "ransac"
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

# evaluate ECC between two frames, return the transformation that aligns im2 to im1 (newer -> older)
def ecc_2frame(
    prev_idx: int,
    curr_idx: int,
    im1: np.ndarray,
    im2: np.ndarray,
    provider: Masking.JumperProvider,
    warp_mode: int = cv2.MOTION_AFFINE,
    number_of_iterations: int = 200, # maximum number of iterations to find the warp matrix
    termination_eps: float = 1e-6, # treshold of convergence
    gaussFiltSize: int = 5,
) -> Tuple[float, np.ndarray]:
    # Convert images to float32 with normalized to [0,1]. it's required by findTransformECC
    im1 = im1.astype(np.float32)/255.0
    im2 = im2.astype(np.float32)/255.0
    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,number_of_iterations,  termination_eps)
    mask1= Masking.create_mask(im1.shape, provider, prev_idx)
    mask2= Masking.create_mask(im2.shape, provider, curr_idx)
    # Combine masks with bitwise AND
    mask = cv2.bitwise_and(mask1, mask2)
    # Initialize warp_matrix before calling the transform finder
    if warp_mode == cv2.MOTION_AFFINE:
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(3, dtype=np.float32)
    # Run the ECC algorithm. The results are stored in warp_matrix.
    # cc is the correlation coefficient
    (cc, warp_matrix) = find_right_transform(im1, im2, warp_mode, criteria, mask, gaussFiltSize, warp_matrix)
    # If the warp mode is AFFINE, then convert the warp matrix to 3x3
    if warp_mode == cv2.MOTION_AFFINE:
        warp_matrix = np.vstack([warp_matrix, [0, 0, 1]]) if warp_mode == cv2.MOTION_AFFINE else warp_matrix
    return cc, warp_matrix



def find_right_transform(
    im1: np.ndarray,
    im2: np.ndarray,
    warp_mode: int,
    criteria,
    inputMask: np.ndarray | None,
    gaussFiltSize: int,
    init_warp: np.ndarray,
    sobel_th: float = 0.03,          # threshold on texture level
    sift_nfeatures: int = 2000,
    ransac_reproj_th: float = 3.0,
    min_inliers: int = 40,
    min_inlier_ratio: float = 0.25,
) -> tuple[float, np.ndarray]:
    valid_mask = inputMask
    # Texture gating (mean Sobel magnitude)
    mean_mag = _mean_sobel_mag(im1, valid_mask)
    if mean_mag < sobel_th:
        return 0.0, _identity_warp(warp_mode)

    # ECC branch
    if str(GLOBAL_VAR).lower() == "ecc":
        try:
            warp = init_warp.copy().astype(np.float32)
            cc, warp = cv2.findTransformECC(
                im1, im2,
                warp,
                warp_mode,
                criteria,
                valid_mask,
                gaussFiltSize
            )
            return float(cc), warp
        except cv2.error:
            return 0.0, _identity_warp(warp_mode)

    # SIFT + RANSAC branch
    try:
        im1_u8 = _to_u8(im1)
        im2_u8 = _to_u8(im2)

        # Detect and compute SIFT descriptors
        sift = cv2.SIFT_create(nfeatures=sift_nfeatures)
        kp1, des1 = sift.detectAndCompute(im1_u8, valid_mask)
        kp2, des2 = sift.detectAndCompute(im2_u8, valid_mask)
        if des1 is None or des2 is None or len(kp1) < 6 or len(kp2) < 6:
            return 0.0, _identity_warp(warp_mode)

        # Match descriptors (L2 + ratio test)
        # it's used to find feasible correspondences between the two images
        # L2 is used because SIFT uses L2 distance that is Euclidean distance in descriptor space 
        bf = cv2.BFMatcher(cv2.NORM_L2)
        # knn works by finding the k best matches for each descriptor from des1 to des2
        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        # m is the best match, n is the second best match
        for m, n in matches:
            # Apply ratio test: if the distance of the best match is significantly lower than the second best, it's a good match
            if m.distance < 0.75 * n.distance:
                good.append(m)
        if len(good) < 6:
            return 0.0, _identity_warp(warp_mode)

        # Convert matches to point arrays
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

        # Estimate affine (or homography) mapping im2 -> im1 (newer -> older)
        if warp_mode == cv2.MOTION_AFFINE:
            A, inliers = cv2.estimateAffinePartial2D(
                pts2, pts1,
                method=cv2.RANSAC,
                ransacReprojThreshold=ransac_reproj_th,
                maxIters=2000,
                confidence=0.99,
                refineIters=10
            )
        else:
            A, inliers = cv2.findHomography(
                pts2, pts1,
                method=cv2.RANSAC,
                ransacReprojThreshold=ransac_reproj_th,
                maxIters=2000,
                confidence=0.99,
                refineIters=10
            )

        if A is None or inliers is None:
            return 0.0, _identity_warp(warp_mode)

        inliers = inliers.ravel().astype(bool)
        n_in = int(inliers.sum())
        n_all = int(inliers.size)
        rin = float(n_in) / float(n_all) if n_all else 0.0

        if n_in < min_inliers or rin < min_inlier_ratio:
            return 0.0, _identity_warp(warp_mode)

        return rin, A.astype(np.float32)

    except cv2.error:
        return 0.0, _identity_warp(warp_mode)


        


# Detect RANSAC/ECC motion over a sequence of frames from the video capture.
# Returns a list of Edge objects.
# Produces multiple constraints:
# short edge: (curr, prev)  (consecutive frames)
# long  edge: (curr, old)  (lookback frames)
def detect_motion_sequence(
    cap,
    provider: "Masking.JumperProvider",
    warp_mode: int = cv2.MOTION_AFFINE,
  
) -> List["Edge"]:

    frame_iter = iter_frames(cap)

    try:
        prev_idx, prev = next(frame_iter)
    except StopIteration:
        return []

    results: List["Edge"] = []

    # Keep at most (lookback + 1) frames: oldest will be lookback frames behind current
    history = deque([(prev_idx, prev)], maxlen=max(2, LOOKBACK + 1))

    for curr_idx, curr in frame_iter:
        # short edge curr -> prev
        try:
            cc, warp_matrix = ecc_2frame(
                prev_idx,
                curr_idx,
                prev,
                curr,
                provider,
                warp_mode=warp_mode,
            )
            results.append(Edge(
                frame_i_index=curr_idx,
                frame_j_index=prev_idx,
                warp_matrix=warp_matrix,
                weight=cc,
            ))
        except cv2.error as e:
            print(f"ECC alignment failed (short) between {prev_idx} and {curr_idx}: {e}")

        # Update history AFTER processing short edge
        history.append((curr_idx, curr))

        # long edge curr -> oldest (only when buffer is full)
        if LOOKBACK > 0 and len(history) == history.maxlen:
            old_idx, old = history[0]

            # avoid duplicating the short edge when lookback==1
            if old_idx != prev_idx:
                try:
                    cc2, warp_matrix2 = ecc_2frame(
                        old_idx,
                        curr_idx,
                        old,
                        curr,
                        provider,
                        warp_mode=warp_mode,
                    )
                    results.append(Edge(
                        frame_i_index=curr_idx,
                        frame_j_index=old_idx,
                        warp_matrix=warp_matrix2,
                        weight=cc2,
                    ))
                except cv2.error as e:
                    print(f"ECC alignment failed (long) between {old_idx} and {curr_idx}: {e}")

        # Advance
        prev_idx, prev = curr_idx, curr

    return results


#-----------------------------------------------------------------------------
#HELPER METHODS
#-----------------------------------------------------------------------------
def compute_ecc_impr(
    prev_gray: np.ndarray,
    cur_gray: np.ndarray,
    warp_matrix: np.ndarray,
    valid_mask: np.ndarray,
    blur_ksize: int = 5,
) -> float:
    # ecc_impr = (E_id - E_ecc) / E_id
    # where E_id  = mean(|prev - cur|) on valid pixels
    #       E_ecc = mean(|prev - warp(cur)|) on valid pixels
    #
    # prev_gray, cur_gray: grayscale uint8 (HxW)
    # warp_matrix: 2x3 affine mapping newer -> older (cur -> prev)
    # valid_mask: uint8 (HxW), 255=valid, 0=invalid
    if prev_gray.ndim != 2 or cur_gray.ndim != 2:
        raise ValueError("prev_gray and cur_gray must be grayscale (HxW)")

    if valid_mask is None:
        valid_mask = np.ones_like(prev_gray, dtype=np.uint8) * 255

    m = (valid_mask > 0)
    if int(m.sum()) == 0:
        return float("nan")

    # ksize must be odd and > 0
    k = int(blur_ksize)
    if k < 1:
        k = 1
    if k % 2 == 0:
        k += 1

    prev_b = cv2.GaussianBlur(prev_gray, (k, k), 0)
    cur_b  = cv2.GaussianBlur(cur_gray,  (k, k), 0)

    # Error with identity
    diff_id = np.abs(prev_b.astype(np.int16) - cur_b.astype(np.int16))
    E_id = float(diff_id[m].mean())

    if not np.isfinite(E_id) or E_id <= 1e-6:
        return float("nan")

    H, W = cur_gray.shape[:2]

    # Apply warp to current (newer) image
    if warp_matrix.shape == (2, 3):
        cur_w = cv2.warpAffine(
            cur_b, warp_matrix, (W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )
    else:
        raise ValueError("warp_matrix must be 2x3 affine for this function")

    diff_e = np.abs(prev_b.astype(np.int16) - cur_w.astype(np.int16))
    E_ecc = float(diff_e[m].mean())

    if not np.isfinite(E_ecc):
        return float("nan")

    return float((E_id - E_ecc) / E_id)


def clamp01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))

# Return an identity warp compatible with the chosen warp_mode
def _identity_warp(warp_mode: int) -> np.ndarray:
    # MOTION_HOMOGRAPHY needs 3×3, others need 2×3
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        return np.eye(3, dtype=np.float32)
    else:
        return np.eye(2, 3, dtype=np.float32)


# Cast float [0 .. 1] to uint8 [0 .. 255] (or pass-through if already uint8)
def _to_u8(im01: np.ndarray) -> np.ndarray:
    if im01.dtype == np.uint8:
        return im01
    return (np.clip(im01, 0.0, 1.0) * 255.0).astype(np.uint8)


# Mean Sobel magnitude computed only on valid pixels
def _mean_sobel_mag(im01: np.ndarray, valid_mask: np.ndarray | None) -> float:
    sobelX = cv2.Sobel(im01, cv2.CV_32F, 1, 0, ksize=3)
    sobelY = cv2.Sobel(im01, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(sobelX, sobelY)
    if valid_mask is None:
        return float(np.mean(np.abs(mag)))
    m = valid_mask > 0
    if int(m.sum()) == 0:
        return 0.0
    return float(np.mean(np.abs(mag[m])))

