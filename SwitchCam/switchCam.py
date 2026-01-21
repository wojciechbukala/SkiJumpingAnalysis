from dataclasses import dataclass
from typing import Iterable, List, Tuple

import cv2

DEFAULT_DIFF_THRESHOLD = 0.35
DEFAULT_MIN_SCENE_LEN = 10
DEFAULT_SAMPLE_RATE = 1
DEFAULT_RESIZE_WIDTH = 320
DEFAULT_HIST_BINS = 32


@dataclass
class ShotChange:
    # attributes for a single detected cut.
    frame_idx: int
    time_s: float
    score: float


def iter_frames(cap, sample_rate: int = DEFAULT_SAMPLE_RATE) -> Iterable[Tuple[int, "cv2.Mat"]]:
    # Yield every Nth frame to control processing load.
    if sample_rate < 1:
        raise ValueError("sample_rate must be >= 1.")

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if sample_rate == 1 or frame_idx % sample_rate == 0:
            yield frame_idx, frame
        frame_idx += 1


def _resize_frame(frame, resize_width: int | None) -> "cv2.Mat":
    # Downscale for faster histogram computation.
    if resize_width is None:
        return frame

    height, width = frame.shape[:2]
    if width <= resize_width:
        return frame

    scale = resize_width / float(width)
    new_height = max(1, int(height * scale))
    return cv2.resize(frame, (resize_width, new_height), interpolation=cv2.INTER_AREA)


def _compute_histogram(frame, bins: int) -> "cv2.Mat":
    # HSV histogram captures color distribution robustly to lighting.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [bins, bins], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist


def detect_shot_changes(
    cap,
    fps: float,
    diff_threshold: float = DEFAULT_DIFF_THRESHOLD,
    min_scene_len: int = DEFAULT_MIN_SCENE_LEN,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    resize_width: int | None = DEFAULT_RESIZE_WIDTH,
    hist_bins: int = DEFAULT_HIST_BINS,
) -> List[ShotChange]:
    # Compare consecutive histograms; large distances indicate a cut.
    if min_scene_len < 1:
        raise ValueError("min_scene_len must be >= 1.")

    changes: List[ShotChange] = []
    prev_hist = None
    last_cut = -min_scene_len

    for frame_idx, frame in iter_frames(cap, sample_rate=sample_rate):
        frame = _resize_frame(frame, resize_width)
        hist = _compute_histogram(frame, hist_bins)

        if prev_hist is not None:
            score = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            if score >= diff_threshold and (frame_idx - last_cut) >= min_scene_len:
                # Enforce a minimum scene length to avoid bursty detections.
                time_s = (frame_idx / fps) if fps else 0.0
                changes.append(ShotChange(frame_idx=frame_idx, time_s=time_s, score=score))
                last_cut = frame_idx

        prev_hist = hist

    return changes


def format_changes(changes: List[ShotChange]) -> str:
    # Human-readable output for console inspection.
    if not changes:
        return "No shot changes detected."
    lines = []
    for change in changes:
        lines.append(
            f"frame={change.frame_idx} time_s={change.time_s:.2f} score={change.score:.3f}"
        )
    return "\n".join(lines)
