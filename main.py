import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import cv2
from SwitchCam import switchCam as shotChange
from ECC.ecc import EccResult 
import ECC.ecc
import jumperTrajectory.trajectory as tj
import matplotlib.pyplot as plt

"""
SkiJumpingAnalysis - main skeleton

File: main.py
"""
__version__ = "0.0.1"

def plot_trajectories_2d(trajectory_list: list[list[tuple[int,int,int]]]) -> None:
        if not trajectory_list:
            print("No trajectory to plot.")
        else:
            for idx, traj in enumerate(trajectory_list):
                plt.figure()
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



def main(argv: Optional[list] = None) -> int:
    print(f"SkiJumpingAnalysis {__version__}")


    # Resolve the video path from the project root.
    root_dir = Path(__file__).resolve().parent
    # Set the path to the video file.
    video_path = root_dir / "videos/21.mp4"

    # Open the video and run shot-change detection.
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    changes = shotChange.detect_shot_changes(cap, fps=fps)

    print(f"Video: {video_path}")
    print(f"Detected changes: {len(changes)}")
    print(shotChange.format_changes(changes))

    # Shot-change detection consumes the capture; rewind before running ECC.
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    list_of_warp_matrices: list[EccResult] = ECC.ecc.detect_ecc_motion_sequence(cap)
    refFrame=0
    trajectory_list: list[list[tuple[float,float,float]]] = []
    for shot in range(len(changes)+1):   
        trajectory_list.append(tj.compute_trajectory(list_of_warp_matrices,refFrame,changes[shot].frame_idx))
        if shot<len(changes):
            refFrame=changes[shot].frame_idx
        else:
            refFrame=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1

    # Plot all collected trajectories
    plot_trajectories_2d(trajectory_list)
    cap.release()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())


