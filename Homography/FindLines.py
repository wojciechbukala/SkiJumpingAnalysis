import cv2
import numpy as np
from ultralytics import YOLO

from pathlib import Path
import Calculations

ROOT_DIR = Path(__file__).resolve().parents[1]
SEGMENTATION_MODEL_PATH = ROOT_DIR / "models/best-segmentation.pt"

MODEL = YOLO(SEGMENTATION_MODEL_PATH)

def FindInrunTracks(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_white = np.array([60, 15, 200])
    upper_white = np.array([100, 30, 255])

    # Create mask for white colors in HSV
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Apply morphological operations: opening then closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)  # Opening: remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)  # Closing: fill holes
    
    # Apply edge mask: zero out left 30% and right 10%
    h, w = mask.shape[:2]
    left_cut = int(w * 0.30)      # Left 40%
    right_cut = int(w * 0.15)     # Right 30%
    edge_mask = np.zeros((h, w), dtype=np.uint8)
    edge_mask[:, left_cut:w-right_cut] = 255  # Middle part only
    mask = cv2.bitwise_and(mask, mask, mask=edge_mask)

    # Apply mask to original image
    result = cv2.bitwise_and(frame, frame, mask=mask)
    result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(result_gray, 100, 150)

    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=40, minLineLength=40, maxLineGap=10)
    lines = [l[0] for l in lines] if lines is not None else []

    filtered_lines = Calculations._eliminate_close_lines(lines, dist_threshold=20, angle_threshold_deg=5)

    track_pair = Calculations._find_track_pair(filtered_lines, h, w)

    result = []
    if track_pair:
        for line in track_pair:
            x1, y1, x2, y2 = int(line[0]), int(line[1]), int(line[2]), int(line[3])
            if x2 != x1:
                a = (y2-y1)/(x2-x1)
                b = y1 - a* x1

                result.append([a, b])

    return result


def FindInrunSteps(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, w = frame.shape[:2]

    roi_y_start, roi_y_end = h // 2, h
    roi_x_start, roi_x_end = w*3 // 5, w

    roi = hsv[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

    lower_white = np.array([80, 30, 50])
    upper_white = np.array([100, 55, 100])
    mask = cv2.inRange(roi, lower_white, upper_white)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 1))
    general_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, horizontal_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, general_kernel, iterations=2)

    edges = cv2.Canny(mask, 50, 100)

    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=20, minLineLength=50, maxLineGap=3)
   
    lines = [l[0] for l in lines] if lines is not None else []

    filtered_lines = Calculations._eliminate_close_lines(lines, dist_threshold=10, angle_threshold_deg=5)
    filtered_lines = [l for l in filtered_lines if abs(Calculations._angle(l)) < 0.1]

    pair = sorted(filtered_lines, key=Calculations._length, reverse=True)[:2]

    result = []
    if len(pair) >= 2:
        for line in pair:
            x1, y1, x2, y2 = line
            gx1, gy1 = x1 + roi_x_start, y1 + roi_y_start
            gx2, gy2 = x2 + roi_x_start, y2 + roi_y_start

        if gx2 != gx1:
            a = (gy2 - gy1) / (gx2 - gx1)
            b = gy1 - a * gx1   
        else:
            a = 0
            b = gy1

        result.append([a, b])
    return result

def FindKHS(frame):
    results = MODEL(source=frame, conf=0.5)
    res = results[0]

    h, w = frame.shape[:2]

    result = []
    for i, mask in enumerate(res.masks.xy):
        points = np.array(mask, dtype=np.float32)
        [vx, vy, x0, y0] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x0, y0 = float(vx[0]), float(vy[0]), float(x0[0]), float(y0[0])
        
        a = vy / vx
        b = y0 - (a*x0)

        result.append([a ,b])

    return result

if __name__ == "__main__":
    pass