import cv2
import numpy as np
from pathlib import Path
import math


def _angle(l):
    x1,y1,x2,y2 = l
    return math.atan2(y2-y1, x2-x1)

def _mid(l):
    x1,y1,x2,y2 = l
    return ((x1+x2)/2.0, (y1+y2)/2.0)

def _dist(a,b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def _angdiff(a,b):
    d = abs(a-b) % math.pi
    return min(d, math.pi-d)

def _length(l):
    x1, y1, x2, y2 = l
    return math.hypot(x2-x1, y2-y1)

def _get_line_points(l, num_points=20):
    x1, y1, x2, y2 = l
    points = []
    for i in range(num_points):
        t = i / (num_points - 1) if num_points > 1 else 0
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        points.append((x, y))
    return points

def _min_dist_between_lines(l1, l2, num_points=20):
    points1 = _get_line_points(l1, num_points)
    points2 = _get_line_points(l2, num_points)
    
    min_dist = float('inf')
    for p1 in points1:
        for p2 in points2:
            dist = _dist(p1, p2)
            if dist < min_dist:
                min_dist = dist
    return min_dist

def eliminate_close_lines(lines, dist_threshold=20, angle_threshold_deg=5):
    lines = [tuple(map(float,l)) for l in lines]
    lines.sort(key=_length, reverse=True)
    angle_threshold = math.radians(angle_threshold_deg)
    kept=[]

    for l in lines:
        close = False
        l_angle = _angle(l)
        for k in kept:
            k_angle = _angle(k)
            if _angdiff(l_angle, k_angle) < angle_threshold and _min_dist_between_lines(l, k) < dist_threshold:
               close = True
               break
        if not close:
            kept.append(l)
    return kept 

def find_track_pair(filtered_lines, img_height, img_width):
    if len(filtered_lines) < 2:
        return None

    sorted_lines = sorted(filtered_lines, key=lambda l: max(l[1], l[3]), reverse=True)

    best_pair = None
    min_score = float('inf')

    for i in range(len(sorted_lines)):
        for j in range(i + 1, len(sorted_lines)):
            l1 = sorted_lines[i]
            l2 = sorted_lines[j]

            ang1, ang2 = _angle(l1), _angle(l2)
            mid1, mid2 = _mid(l1), _mid(l2)
            
            angle_diff = _angdiff(ang1, ang2)
            
            dist_x = abs(mid1[0] - mid2[0])
            
            dist_from_bottom = img_height - max(max(l1[1], l1[3]), max(l2[1], l2[3]))
            
            score = (angle_diff * 100) + (dist_x * 0.5) + (dist_from_bottom * 0.1)
            if dist_x < img_width * 0.25 and angle_diff < math.radians(10):
                if score < min_score:
                    min_score = score
                    best_pair = (l1, l2)
                    print(f'angle_diff: {angle_diff * 100}, dist:{dist_x * 0.5}, dist_from_bottom: {dist_from_bottom * 0.1}')

    return best_pair



def FindLinesVertical(img, debug=False):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

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
    result = cv2.bitwise_and(img, img, mask=mask)
    result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(result_gray, 100, 150)

    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=40, minLineLength=40, maxLineGap=10)
    lines = [l[0] for l in lines] if lines is not None else []
    
    
    filtered_lines = eliminate_close_lines(lines, dist_threshold=20, angle_threshold_deg=5)


    lines_img = img.copy()
    if filtered_lines is not None:
        for x1,y1,x2,y2 in filtered_lines:
             cv2.line(lines_img, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
        print(f"Found {len(lines)} lines")
    else:
        print("No lines detected")

    track_pair = find_track_pair(filtered_lines, h, w)

    if track_pair:
        for line in track_pair:
            x1, y1, x2, y2 = int(line[0]), int(line[1]), int(line[2]), int(line[3])
            if x2 != x1:
                a = (y2 - y1) / (x2 - x1)
                b = y1 - a * x1

                y_left = 0
                x_left = int((y_left-b)/a)
                
                y_right = h
                x_right = int((y_right-b)/a)
            
            cv2.line(lines_img, (x_left, y_left), (x_right, y_right), (0, 0, 255), 2)


    if debug:  
        result = cv2.resize(result, (1000, 600))
        edges = cv2.resize(edges, (1000, 600))
        lines_img = cv2.resize(lines_img, (1000, 600))
        cv2.imshow("Thresholding Result", result)
        cv2.imshow("Edges", edges)
        cv2.imshow("Detected Lines", lines_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return track_pair


def FindLinesHorizontal(img, debug=False):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = img.shape[:2]

    roi_y_start, roi_y_end = h // 2, h
    roi_x_start, roi_x_end = w*3 // 5, w

    roi = hsv[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

    lower_white = np.array([80, 30, 50])
    upper_white = np.array([100, 55, 100])
    mask = cv2.inRange(roi, lower_white, upper_white)
    
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 1))
    general_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, horizontal_kernel)
    ask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, general_kernel, iterations=2)

    edges = cv2.Canny(mask, 50, 100)

    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=20, minLineLength=50, maxLineGap=3)
   
    lines = [l[0] for l in lines] if lines is not None else []

    filtered_lines = eliminate_close_lines(lines, dist_threshold=10, angle_threshold_deg=5)
    filtered_lines = [l for l in filtered_lines if abs(_angle(l)) < 0.1]


    lines_img = img.copy()

    if filtered_lines is not None:
        for line in filtered_lines:
            x1,y1,x2,y2 = line
            full_x1 = int(x1 + roi_x_start)
            full_y1 = int(y1 + roi_y_start)
            full_x2 = int(x2 + roi_x_start)
            full_y2 = int(y2 + roi_y_start)
            
            cv2.line(lines_img, (full_x1, full_y1), (full_x2, full_y2), (0, 255, 0), 2)
        print(f"Found {len(filtered_lines)} lines")
    else:
        print("No lines detected")

    pair = sorted(filtered_lines, key=_length, reverse=True)[:2]
    

    if len(pair) >= 2:
        print(f'Pair: {pair}')
        for line in pair:
            x1, y1, x2, y2 = line
            gx1, gy1 = x1 + roi_x_start, y1 + roi_y_start
            gx2, gy2 = x2 + roi_x_start, y2 + roi_y_start

            if gx2 != gx1:
                a = (gy2 - gy1) / (gx2 - gx1)
                b = gy1 - a * gx1

                x_left = 0
                y_left = int(a * x_left + b)
                
                x_right = w
                y_right = int(a * x_right + b)
            else:
                x_left, y_left = int(gx1), 0
                x_right, y_right = int(gx1), h

            cv2.line(lines_img, (x_left, y_left), (x_right, y_right), (0, 0, 255), 2)


    if debug:
        mask = cv2.resize(mask, (1000, 600))
        edges = cv2.resize(edges, (1000, 600))
        lines_img = cv2.resize(lines_img, (1000, 600))
        cv2.imshow("Mask", mask)
        cv2.imshow("Edges", edges)
        cv2.imshow("Lines", lines_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def inrun_homography():
    pass

def jump_homography():
    pass


if __name__ == '__main__':
    root_dir = Path(__file__).resolve().parents[1]
    img_path = root_dir / 'detectionOutputs/inrun2_out.jpg'

    img = cv2.imread(img_path)

    FindLinesHorizontal(img, debug=True)
    FindLinesVertical(img, debug=True)
