from pathlib import Path
import cv2
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from SwitchCam import switchCam
from FindLines import FindKHS


def _find_landing_frame(camera_intervals):
    jump_interval = next((interval for interval in camera_intervals if interval.camera_type == "Jump"), None)
    
    if jump_interval is not None:
        jump_frame_idx = jump_interval.start.frame_idx + int(((jump_interval.end.frame_idx - jump_interval.start.frame_idx) * 0.85))
        return jump_frame_idx
    else:
        return None

def _find_inrun_frame(camera_intervals):
    inrun_interval = next((interval for interval in camera_intervals if interval.camera_type == "Acceleration"), None)
    if inrun_interval is None:
        inrun_interval = next((interval for interval in camera_intervals if interval.camera_type == "AccelerationA"), None)

    if inrun_interval is not None:
        inrun_frame_idx = inrun_interval.start.frame_idx + int(((inrun_interval.end.frame_idx - inrun_interval.start.frame_idx)* 0.1))
        return inrun_frame_idx
    else:
        return None
    
def _get_frame(video_path, frame_number):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        return frame
    return None

def find_frames(video_path, inrun_frame=True, landing_frame=True):
    # open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    changes = switchCam.detect_shot_changes(cap, fps=fps)
    cap.release()

    # find shot changes intervals
    intervals = switchCam.get_camera_intervals(changes, total_frames, fps)

    # find proper frame for inrun and landing lines detection
    inrun_frame_idx = _find_inrun_frame(intervals)
    landing_frame_idx = _find_landing_frame(intervals)

    inrun_frame = _get_frame(video_path, inrun_frame_idx) if inrun_frame_idx is not None else None
    landing_frame = _get_frame(video_path, landing_frame_idx) if landing_frame_idx is not None else None

    return [inrun_frame, landing_frame]


if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parents[1]
    find_frames(root_dir/"detectionOutputs/G_30out.mp4")