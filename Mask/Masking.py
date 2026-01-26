import numpy as np
import cv2
import pandas as pd

OVERLAY_RECTS: list[tuple[int, int, int, int]] = [
    (0, 39, 538, 296),
    (0, 544, 752, 1046),
]
OVERLAY_RECTS2: list[tuple[int, int, int, int]] = [
    (0, 39, 538, 296),
    (83, 922, 571, 984),
]

def create_mask(
    frame_shape: tuple[int, int],
    prev_idx: int = 0,
    margin: int = 20,
) -> np.ndarray:
    height, width = frame_shape
    mask = np.ones((height, width), dtype=np.uint8) * 255

    skier_bbox = get_bbox(prev_idx)
    if skier_bbox is not None:
        x1, y1, x2, y2 = map(int, skier_bbox)
        x1 = max(x1 - margin, 0)
        y1 = max(y1 - margin, 0)
        x2 = min(x2 + margin, width)
        y2 = min(y2 + margin, height)
        mask[y1:y2, x1:x2] = 0

    # Mask out any overlay rectangles (x1, y1, x2, y2).
    overlay_rects = OVERLAY_RECTS if prev_idx < 150 else OVERLAY_RECTS2
    for (x1, y1, x2, y2) in overlay_rects:
        x1 = max(0, min(int(x1), width))
        y1 = max(0, min(int(y1), height))
        x2 = max(0, min(int(x2), width))
        y2 = max(0, min(int(y2), height))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 0

    return mask


from pathlib import Path
# Resolve the video path from the project root.
root_dir = Path(__file__).resolve().parents[1]
# Set the path to the video file.
csv_path = root_dir / "jumperbox/20trajectory_2d.csv"


df = pd.read_csv(csv_path)  # es: colonne frame,x1,y1,x2,y2
df["frame"] = df["frame"].astype(int)

# Create a dictionary mapping frame indices to bounding boxes
bbox_by_frame = {}
for row in df.itertuples(index=False):
    x1, y1, x2, y2 = float(row.x1), float(row.y1), float(row.x2), float(row.y2)
    bbox_by_frame[int(row.frame)] = np.array([x1, y1, x2, y2], dtype=np.float32)

def get_bbox(frame_idx: int):
    return bbox_by_frame.get(frame_idx, None)
