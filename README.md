# SkiJumpingAnalysis
This is a repository for IACV project 'F11. Visual Analysis of Sport Events' supervised by prof. Vincenzo Caglioti.
The chosen sport for conducting analysis is Ski Jumping.

Ski jumping video samples drive: https://drive.google.com/drive/folders/1NacD2FmqXp9dofT87Ru5cCQNJ2sDAhCa?usp=sharing
folder called videos to put video samples. 

Roboflow annotated data: https://app.roboflow.com/skijumpinglabeling

Presentation: https://docs.google.com/presentation/d/1ppiDII5XNFy0YP4QXc20gLu-Mdj3y_mMC-GaFx9Spqw/edit?usp=sharing

## Geometric reconstruction of the track corridor

The `track_geometry/` package contains the initial implementation of the
geometric model requested for reconstructing the two parallel curves that bound
the ski-jump track corridor.

It starts from two ordered image curves, optional point correspondences, and
optionally the camera calibration matrix `K`. The experiment entry point is:

```bash
python3 experiments/reconstruct_parallel_curves.py \
  --input configs/parallel_curves_example.json \
  --output-dir outputGeometry
```
