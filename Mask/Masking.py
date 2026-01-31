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

class JumperProvider:
    def __init__(self, csv_path: str):
        self.bbox_by_frame = {}
        self.point_by_frame = {}
        self._load_csv(csv_path)

    def _load_csv(self, csv_path: str):
        df = pd.read_csv(csv_path)
        for row in df.itertuples(index=False):
            f_idx = int(row.frame)
            self.bbox_by_frame[f_idx] = np.array([row.x1, row.y1, row.x2, row.y2], dtype=np.float32)
            self.point_by_frame[f_idx] = (float(row.kf_x), float(row.kf_y))

    def get_bbox(self, frame_idx: int):
        return self.bbox_by_frame.get(frame_idx)
    
    def get_point(self, frame_idx: int):
        return self.point_by_frame.get(frame_idx)

def create_mask(
    frame_shape: tuple[int, int],
    provider: JumperProvider,
    frame_idx: int = 0,
    margin: int = 1,
) -> np.ndarray:
    height, width = frame_shape
    # Start with a mask that is all valid (255)
    mask = np.ones((height, width), dtype=np.uint8) * 255
    # put a 0 where the skier is
    skier_bbox = provider.get_bbox(frame_idx)
    if skier_bbox is not None:
        arr = np.asarray(skier_bbox, dtype=np.float32)
        if arr.size >= 4 and np.all(np.isfinite(arr[:4])):
            skier_bbox = arr[:4]
        else:
            skier_bbox = None

    if skier_bbox is not None:
        x1, y1, x2, y2 = map(int, skier_bbox)
        x1 = max(x1 - margin, 0)
        y1 = max(y1 - margin, 0)
        x2 = min(x2 + margin, width)
        y2 = min(y2 + margin, height)
        mask[y1:y2, x1:x2] = 0

    # Mask out any overlay rectangles (x1, y1, x2, y2).
    # put a 0 where there are overlay rectangles
    overlay_rects = OVERLAY_RECTS if frame_idx < 150 else OVERLAY_RECTS2
    for (x1, y1, x2, y2) in overlay_rects:
        x1 = max(0, min(int(x1), width))
        y1 = max(0, min(int(y1), height))
        x2 = max(0, min(int(x2), width))
        y2 = max(0, min(int(y2), height))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 0

    return mask


# from pathlib import Path
# # Resolve the video path from the project root.
# root_dir = Path(__file__).resolve().parents[1]
# # Set the path to the video file.
# csv_path = root_dir / "jumperbox/20trajectory_2d.csv"


# df = pd.read_csv(csv_path)  # es: colonne frame,x1,y1,x2,y2
# df["frame"] = df["frame"].astype(int)

# # Create a dictionary mapping frame indices to bounding boxes
# bbox_by_frame = {}
# for row in df.itertuples(index=False):
#     x1, y1, x2, y2 = float(row.x1), float(row.y1), float(row.x2), float(row.y2)
#     bbox_by_frame[int(row.frame)] = np.array([x1, y1, x2, y2], dtype=np.float32)

# def get_bbox(frame_idx: int):
#     return bbox_by_frame.get(frame_idx, None)
