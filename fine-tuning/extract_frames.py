import cv2
import os
import random

MIN_INTERVAL = 500  # minimum interval in milliseconds
MAX_INTERVAL = 2500  # maximum interval in milliseconds

def extract_frames(input_folder, output_folder, min_interval=MIN_INTERVAL, max_interval=MAX_INTERVAL):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    total_frames = 0
    video_frame_counts = []

    for video_file in os.listdir(input_folder):
        if video_file.endswith(('.mp4', '.avi', '.mov')):
            video_path = os.path.join(input_folder, video_file)
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)

            frame_count = 0
            saved_frame_count = 0
            next_save_frame = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count >= next_save_frame:
                    frame_filename = f"{os.path.splitext(video_file)[0]}_frame_{saved_frame_count:05d}.jpg"
                    cv2.imwrite(os.path.join(output_folder, frame_filename), frame)
                    saved_frame_count += 1

                    # Losuj następny odstęp w klatkach
                    interval_ms = random.randint(min_interval, max_interval)
                    interval_frames = max(1, int(fps * (interval_ms / 1000.0)))
                    next_save_frame = frame_count + interval_frames

                frame_count += 1

            cap.release()
            total_frames += saved_frame_count
            video_frame_counts.append(saved_frame_count)
            print(f"Extracted {video_file}: {saved_frame_count} frames")

    return total_frames, video_frame_counts

if __name__ == "__main__":
    input_folder = "videos"
    output_folder = "extracted_frames"
    total_frames, video_frame_counts = extract_frames(input_folder, output_folder)
    
    num_videos = len(video_frame_counts)
    if num_videos > 0:
        average_frames = total_frames / num_videos
        print(f"\nSummary:")
        print(f"Total generated frames: {total_frames}")
        print(f"Average frames per video: {average_frames:.2f}")
    else:
        print("No videos to process.")