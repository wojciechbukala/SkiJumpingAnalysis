from pathlib import Path
import cv2
import matplotlib
from src.SwitchCam import switchCam as shotChange
from src.SwitchCam.switchCam import ShotChange
from src.transform.graphtr import Edge
import src.transform.graphtr as graph
import src.transform.transform as affineTransform
import src.jumperTrajectory.trajectory as tj
from src.Mask.Masking import JumperProvider
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from typing import List
import src.jumperTrajectory.trajectory as tj
from src.jumperTrajectory.smoothing import smooth_trajectory
"""
SkiJumpingAnalysis - main skeleton

File: main.py
"""
__version__ = "0.0.5"
# CHANGE HERE TO SET THE DESIRED MOTION MODEL !!!!!!!!!!!!!!!!!!!!!
WARP_MODE = cv2.MOTION_AFFINE
VIDEO_PATH = Path("detectionOutputs/O_20out.mp4")
CSV_PATH = Path("detectionOutputs/O_20trajectory.csv")


def plot_trajectories_2d(
    trajectory_list: list[list[tuple[int, int, int]]],
    output_dir: Path,
) -> None:
        if not trajectory_list:
            print("No trajectory to plot.")
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        for idx, traj in enumerate(trajectory_list):
            plt.figure(figsize=(10, 6))
            if not traj:
                plt.title(f"Trajectory {idx} (empty)")
                plt.savefig(output_dir / f"trajectory_{idx:02d}.png", dpi=150)
                plt.close()
                continue
            xs = [p[0] for p in traj]
            ys = [p[1] for p in traj]
            plt.plot(xs, ys, marker='o')
            plt.gca().invert_yaxis()
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f'Trajectory {idx}')
            plt.grid(True)
            plt.savefig(output_dir / f"trajectory_{idx:02d}.png", dpi=150)
            plt.close()

def _segment_edges(edges: list[Edge], start: int, stop: int) -> list[Edge]:
    """Return edges within [start, stop) reindexed to start at 0."""
    seg: list[Edge] = []
    for e in edges:
        if start <= e.frame_i_index < stop and start <= e.frame_j_index < stop:
            seg.append(
                Edge(
                    frame_i_index=e.frame_i_index - start,
                    frame_j_index=e.frame_j_index - start,
                    warp_matrix=e.warp_matrix,
                    weight=e.weight,
                )
            )
    return seg



def main() -> int:
    print(f"SkiJumpingAnalysis {__version__}")

    # Load csv data
    if not CSV_PATH.exists():
        print(f"CSV file not found: {CSV_PATH}")
        return 1
    provider = JumperProvider(str(CSV_PATH))

    # Open the video and run shot-change detection.
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"Could not open video: {VIDEO_PATH}")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    changes = shotChange.detect_shot_changes(cap, fps=fps)

    print(f"Video: {VIDEO_PATH}")
    print(f"Detected changes: {len(changes)}")
    print(shotChange.format_changes(changes))

    # Shot-change detection consumes the capture; rewind before running ECC/RANSAC.
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    list_of_warp_matrices: list[Edge] = affineTransform.detect_motion_sequence(cap, provider, WARP_MODE)
    refFrame=0
    endFrame=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1
    trajectory_list_graph: list[list[tuple[float,float,float]]] = []
    trajectory_list_cum: list[list[tuple[float,float,float]]] = []
    for shot in range(len(changes)+1): 
        if shot < len(changes):
            stop_frame = changes[shot].frame_idx
        else:
            stop_frame = endFrame+1

        num_frame = stop_frame - refFrame
        if num_frame <= 0:
            refFrame = stop_frame
            continue

        seg_edges = _segment_edges(list_of_warp_matrices, refFrame, stop_frame)
        transformations = graph.solve_graph(
            num_frame,
            seg_edges,
            ref=0,
            index_offset=refFrame,
        )
        trajectory_list_graph.append(
            tj.compute_trajectory(transformations, refFrame, stop_frame, provider)
        )
        trajectory_list_cum.append(
            tj.compute_trajectory(seg_edges, refFrame, stop_frame, provider)
        )

        refFrame = stop_frame

    # Apply smoothing to all detected trajectories (Post-Processing)
    # We process each shot independently to avoid artifacts at camera switches
    trajectory_list_graph_smooth = [smooth_trajectory(t) for t in trajectory_list_graph]
    trajectory_list_cum_smooth = [smooth_trajectory(t) for t in trajectory_list_cum]

    # Plot all collected trajectories
    plot_trajectories_2d(trajectory_list_graph, Path("outputPlot") / "graph")
    plot_trajectories_2d(trajectory_list_cum, Path("outputPlot") / "cumulative")
    # Plot the smoothed trajectories
    plot_trajectories_2d(trajectory_list_graph_smooth, Path("outputPlot") / "graph_smooth")
    plot_trajectories_2d(trajectory_list_cum_smooth, Path("outputPlot") / "cumulative_smooth")
    cap.release()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

