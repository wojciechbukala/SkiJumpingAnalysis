from pathlib import Path
import sys
import cv2
import os
import switchCam as shotChange

def visualize_intervals(video_path, intervals, output_path=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define output video if needed
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    intervals_mapped = [
        (interval.start.frame_idx, interval.end.frame_idx, interval.camera_id)
        for interval in intervals
    ]
        
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        camera_id = None
        for start_idx, end_idx, cam_id in intervals_mapped:
            if start_idx <= frame_idx <= end_idx:
                camera_id = cam_id
                break
        
        if camera_id is not None:
            cv2.putText(frame, f"Camera {camera_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        if output_path:
            out.write(frame)
        
        frame_idx += 1
    
    cap.release()
    if output_path:
        out.release()
        print(f"Visualization saved to: {output_path}")

def find_landing_frame(camera_intervals):
    jump_interval = next((interval for interval in camera_intervals if interval.camera_type == "Jump"), None)
    
    if jump_interval is not None:
        jump_frame_idx = jump_interval.start.frame_idx + int(((jump_interval.end.frame_idx - jump_interval.start.frame_idx) * 0.83))
        return jump_frame_idx
    else:
        return None

def find_inrun_frame(camera_intervals):
    inrun_interval = next((interval for interval in camera_intervals if interval.camera_type == "Acceleration"), None)
    if inrun_interval is None:
        inrun_interval = next((interval for interval in camera_intervals if interval.camera_type == "AccelerationA"), None)

    if inrun_interval is not None:
        inrun_frame_idx = inrun_interval.start.frame_idx + int(((inrun_interval.end.frame_idx - inrun_interval.start.frame_idx)* 0.1))
        return inrun_frame_idx
    else:
        return None
    

def main() -> int:
    # Resolve the video path from the project root.
    root_dir = Path(__file__).resolve().parents[1]
    # Set the path to the video file.
    video_path = root_dir / "detectionOutputs/G_30out.mp4"

    # Open the video and run shot-change detection.
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #changes is a list of ShotChange objects it has frame_idx,time_s,score
    changes = shotChange.detect_shot_changes(cap, fps=fps)
    cap.release()

    intervals = shotChange.get_camera_intervals(changes, total_frames, fps)

    print(f"Video: {video_path}")
    print(f"Detected changes: {len(changes)}")
    print(shotChange.format_changes(changes))
    print("\nCamera Intervals:")
    print(shotChange.format_camera_intervals(intervals))

    landing_frame = find_landing_frame(intervals)
    inrun_frame = find_inrun_frame(intervals)
    
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
