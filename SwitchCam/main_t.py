from pathlib import Path
import sys
import cv2
import os
import switchCam as shotChange

def main() -> int:
    # Resolve the video path from the project root.
    root_dir = Path(__file__).resolve().parents[1]
    video_path = root_dir / "videos/20.mp4"

    # Open the video and run shot-change detection.
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    changes = shotChange.detect_shot_changes(cap, fps=fps)
    cap.release()

    print(f"Video: {video_path}")
    print(f"Detected changes: {len(changes)}")
    print(shotChange.format_changes(changes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
