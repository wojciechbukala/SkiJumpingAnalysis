import numpy as np
import cv2
import pandas as pd
OVERLAY_RECTS: list[tuple[int,int,int,int]] = [
    (0,39,538,296)
    (0,544,752,1046)
]
OVERLAY_RECTS2: list[tuple[int,int,int,int]] = [
    (0,39,538,296)
    (83,922,571,984)
]

def create_mask(frame_shape: tuple[int, int],
                 prev_idx: int =0  ,  
                margin:int=20,

                ) -> np.ndarray: 
        high, width = frame_shape 
        mask=np.ones((high, width), dtype=np.uint8)*255
        skier_bbox = get_bbox(prev_idx)
    # Mask out the skier region if bbox is provided
        if skier_bbox is not None:
            x, y, w, h = skier_bbox
            x1 = max(x - margin, 0)
            y1 = max(y - margin, 0)
            x2 = min(x + w + margin, width)
            y2 = min(y + h + margin, high)
            mask[y1:y2, x1:x2] = 0
            #if time<5seconds
        ovr = OVERLAY_RECTS if prev_idx < 150 else OVERLAY_RECTS2
    # Mask out any overlay rectangles if provided
        if ovr is not None:
            for (x, y, w, h) in ovr:
                mask[y:y+h, x:x+w] = 0
        return mask


from pathlib import Path
# Resolve the video path from the project root.
root_dir = Path(__file__).resolve().parents[1]
# Set the path to the video file.
csv_path = root_dir / "jumperbox/20.csv"


df = pd.read_csv(csv_path)  # es: colonne frame,x1,y1,x2,y2
df["frame"] = df["frame"].astype(int)

# Create a dictionary mapping frame indices to bounding boxes
bbox_by_frame = {}
for row in df.itertuples(index=False):
    x1, y1, x2, y2 = float(row.x1), float(row.y1), float(row.x2), float(row.y2)
    bbox_by_frame[int(row.frame)] = np.array([x1, y1, x2, y2], dtype=np.float32)

def get_bbox(frame_idx: int):
    return bbox_by_frame.get(frame_idx, None)
