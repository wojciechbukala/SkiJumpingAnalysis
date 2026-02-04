import math
import numpy as np
import cv2

# calculate the angle of line l wrt image
def _angle(l):
    x1,y1,x2,y2 = l
    return math.atan2(y2-y1, x2-x1)

# calculate the mid point of line l
def _mid(l):
    x1,y1,x2,y2 = l
    return ((x1+x2)/2.0, (y1+y2)/2.0)

# calculate distance between a and b
def _dist(a,b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

# calculate the angle difference between a and b
def _angdiff(a,b):
    d = abs(a-b) % math.pi
    return min(d, math.pi-d)

# caluculate length of the line l
def _length(l):
    x1, y1, x2, y2 = l
    return math.hypot(x2-x1, y2-y1)

# get constantly ditributed points among line
def _get_line_points(l, num_points=20):
    x1, y1, x2, y2 = l
    points = []
    for i in range(num_points):
        t = i / (num_points - 1) if num_points > 1 else 0
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        points.append((x, y))
    return points

# get distance between lines calcualted with distributed points
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

# eliminate lines that are very close to each other, pick the longer one
# lines close to each other are often the same orginal line/object
def _eliminate_close_lines(lines, dist_threshold=20, angle_threshold_deg=5):
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

# find the best pair for inrun track lines
def _find_track_pair(filtered_lines, img_height, img_width):
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