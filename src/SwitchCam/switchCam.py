from dataclasses import dataclass
from typing import Iterable, List, Tuple

import cv2

DEFAULT_DIFF_THRESHOLD = 0.40
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

@dataclass
class Interval:
    frame_idx: int
    time_s: float


@dataclass
class CameraInterval:
    camera_id: int
    start: Interval
    end: Interval
    camera_type: str = "unknown"


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
) -> Tuple[List[ShotChange], int]:
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



def get_camera_intervals(
    changes: List[ShotChange],
    total_frames: int,
    fps: float,
    min_frames_to_end: int = 15
) -> List[CameraInterval]:
    if not changes:
        return [
            CameraInterval(
                camera_id=0,
                start=Interval(frame_idx=0, time_s=0.0),
                end=Interval(frame_idx=total_frames - 1, time_s=(total_frames - 1) / fps if fps else 0.0)
            )
        ]
    
    # Create list of all boundary points (start, changes, end)
    boundaries = [Interval(frame_idx=0, time_s=0.0)]
    boundaries.extend([Interval(frame_idx=c.frame_idx, time_s=c.time_s) for c in changes])
    boundaries.append(Interval(frame_idx=total_frames - 1, time_s=(total_frames - 1) / fps if fps else 0.0))
    
    # Create intervals between consecutive boundaries without overlap
    intervals = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        next_boundary = boundaries[i + 1]
        
        # End frame is one frame before the next boundary starts
        end_frame_idx = next_boundary.frame_idx - 1
        end_time_s = (end_frame_idx / fps) if fps else 0.0
        
        intervals.append(CameraInterval(
            camera_id=i,
            start=start,
            end=Interval(frame_idx=end_frame_idx, time_s=end_time_s)
        ))
    
    if intervals[-1].end.frame_idx - intervals[-1].start.frame_idx < 15:
        intervals.pop()

    if (
        len(intervals) == 3
        and intervals[0].end.time_s - intervals[0].start.time_s > 2
        and intervals[0].end.time_s - intervals[0].start.time_s < 6
        and intervals[1].end.time_s - intervals[1].start.time_s > 2.5
        and intervals[1].end.time_s - intervals[1].start.time_s < 6
        and intervals[2].end.time_s - intervals[2].start.time_s > 3.5
        and intervals[2].end.time_s - intervals[2].start.time_s < 6
    ):
        intervals[0].camera_type = 'Start'
        intervals[1].camera_type = 'Acceleration'
        intervals[2].camera_type = 'Jump'
    elif (
        len(intervals) == 4
        and intervals[0].end.time_s - intervals[0].start.time_s > 2
        and intervals[0].end.time_s - intervals[0].start.time_s < 6
        and intervals[2].end.time_s - intervals[1].start.time_s > 2.5
        and intervals[2].end.time_s - intervals[1].start.time_s < 6
        and intervals[3].end.time_s - intervals[3].start.time_s > 3.5
        and intervals[3].end.time_s - intervals[3].start.time_s < 6
    ):
        intervals[0].camera_type = 'Start'
        intervals[1].camera_type = 'AccelerationA'
        intervals[2].camera_type = 'AccelerationB'
        intervals[3].camera_type = 'Jump'


    return intervals

def name_cameras(intervals: List[CameraInterval]) -> List[str]:
    names = []
    for interval in intervals:
        names.append(f"Camera {interval.camera_id}")
    return names


def format_camera_intervals(intervals: List[CameraInterval]) -> str:
    if not intervals:
        return "No intervals."
    lines = []
    for interval in intervals:
        lines.append(
            f"Camera {interval.camera_type} ({interval.camera_id}): frame {interval.start.frame_idx}-{interval.end.frame_idx} "
            f"({interval.start.time_s:.2f}s - {interval.end.time_s:.2f}s)"
        )
    return "\n".join(lines)