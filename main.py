from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import cv2
from SwitchCam import switchCam as shotChange
from transform.transform import EccResult 
import transform.transform as affineTransform
import jumperTrajectory.trajectory as tj
from Mask.Masking import JumperProvider
import matplotlib.pyplot as plt

"""
SkiJumpingAnalysis - main skeleton

File: main.py
"""
__version__ = "0.0.2"

def plot_trajectories_2d(trajectory_list: list[list[tuple[int,int,int]]]) -> None:
        if not trajectory_list:
            print("No trajectory to plot.")
            return

        for idx, traj in enumerate(trajectory_list):
            plt.figure(figsize=(10, 6))
            if not traj:
                plt.title(f"Trajectory {idx} (empty)")
                plt.show()
                continue
            xs = [p[0] for p in traj]
            ys = [p[1] for p in traj]
            plt.plot(xs, ys, marker='o')
            plt.gca().invert_yaxis()
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f'Trajectory {idx}')
            plt.grid(True)
            plt.show()



def main(video_path: Path, csv_path: Path) -> int:
    print(f"SkiJumpingAnalysis {__version__}")

    # Load csv data
    if not csv_path.exists():
        print(f"CSV file not found: {csv_path}")
        return 1
    provider = JumperProvider(str(csv_path))

    # Open the video and run shot-change detection.
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    changes = shotChange.detect_shot_changes(cap, fps=fps)

    print(f"Video: {video_path}")
    print(f"Detected changes: {len(changes)}")
    print(shotChange.format_changes(changes))

    # Shot-change detection consumes the capture; rewind before running ECC/RANSAC.
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    list_of_warp_matrices: list[EccResult] = affineTransform.detect_ecc_motion_sequence(cap, provider)
    refFrame=0
    endFrame=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1
    trajectory_list: list[list[tuple[float,float,float]]] = []
    for shot in range(len(changes)+1): 

        if shot<len(changes): 
            trajectory_list.append(tj.compute_trajectory(list_of_warp_matrices,refFrame,changes[shot].frame_idx, provider))
 
            refFrame=changes[shot].frame_idx
        else:
            trajectory_list.append(tj.compute_trajectory(list_of_warp_matrices,refFrame,endFrame, provider))

    # Plot all collected trajectories
    plot_trajectories_2d(trajectory_list)
    cap.release()
    return 0

if __name__ == "__main__":
    VIDEO_PATH = Path("detectionOutputs/G_30out.mp4")
    CSV_PATH = Path("detectionOutputs/G_30trajectory.csv")

    raise SystemExit(main(VIDEO_PATH, CSV_PATH))


