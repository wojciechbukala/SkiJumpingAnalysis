from pathlib import Path
import cv2
from FindFrame import find_frames
from FindLines import FindInrunTracks, FindInrunSteps, FindKHS

ROOT_DIR = Path(__file__).resolve().parents[1]

def draw_lines(frame, params):

    h, w = frame.shape[:2]

    for a, b in params:

        points = []

        # x = 0
        y = a * 0 + b
        if 0 <= y < h:
            points.append((0, int(y)))

        # x = w-1
        y = a * (w - 1) + b
        if 0 <= y < h:
            points.append((w - 1, int(y)))

        # y = 0
        if abs(a) > 1e-6:
            x = (0 - b) / a
            if 0 <= x < w:
                points.append((int(x), 0))

        # y = h-1
        if abs(a) > 1e-6:
            x = ((h - 1) - b) / a
            if 0 <= x < w:
                points.append((int(x), h - 1))

        # draw line
        if len(points) >= 2:
            cv2.line(frame, points[0], points[1], (0, 0, 255), 2)

    frame = cv2.resize(frame, (1000, 600))
    cv2.imshow("Line", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def pipeline(video_path, inrun=True, landing=True, debug=False):
    inrun_frame, landing_frame = find_frames(ROOT_DIR/video_path, inrun_frame=inrun, landing_frame=landing)
    
    if inrun_frame is not None:
        track_lines = FindInrunTracks(inrun_frame)
        print(track_lines)

        if debug:
            draw_lines(inrun_frame, track_lines)

        stairs_lines = FindInrunSteps(inrun_frame)
        print(stairs_lines)

        if debug:
            draw_lines(inrun_frame, stairs_lines)

    else:
        print('Feasable frame for inrun not found')

    if landing_frame is not None:
        lines = FindKHS(landing_frame)

        if debug:
            draw_lines(landing_frame, lines)
    else:
        print('Feasable frame for landing not found')

if __name__ == '__main__':
    pipeline("detectionOutputs/O_30out.mp4", inrun=True, landing=True, debug=True)
