# SkiJumpingAnalysis

Repository for the IACV project "F11. Visual Analysis of Sport Events", supervised by prof. Vincenzo Caglioti.
The selected sport is ski jumping.

## Repository layout

- `video_frame_picker.py`: selects and saves the reference frame used for calibration and slope reconstruction.
- `feature_extraction.py`: detects or manually selects image lines and writes a numbered preview.
- `calibration_phase.py`: estimates the camera calibration from grouped parallel lines.
- `main.py`: runs the main pipeline steps: calibration, slope reconstruction, slope visualization, and jumper trajectory lifting.
- `jumper_trajectory.py`: configuration and implementation entry point for the jumper trajectory stage.
- `outputGeometry/`: generated calibration, reconstruction, and trajectory outputs.
- `detectionOutputs/`: expected folder for detection videos and trajectory CSV files.
- `report/`: LaTeX report source and generated PDF.

## Python setup

## Input files

The current configuration expects these files:

```text
detectionOutputs/G_30out.mp4
detectionOutputs/G_30trajectory.csv
```

If you use a different video, update these constants:

- `video_frame_picker.py`: `VIDEO_PATH`
- `jumper_trajectory.py`: `VIDEO_PATH`, `CSV_PATH`, `REFERENCE_FRAME`
- `main.py`: `JUMPER_TRAJECTORY_CAMERA_ID` if the target camera interval changes

Video samples are available here:
https://drive.google.com/drive/folders/1NacD2FmqXp9dofT87Ru5cCQNJ2sDAhCa?usp=sharing

Put downloaded or generated videos in `detectionOutputs/`.

## Pipeline instructions

1. Select the reference frame:

```bash
python video_frame_picker.py
```

2. Extract calibration lines:

```bash
python feature_extraction.py
```

Choose `auto` for automatic line detection or `manuale` for manual line selection.

Outputs:

```text
outputGeometry/calibration_lines_preview.jpg
outputGeometry/calibration_edges.jpg
outputGeometry/calibration_lines.csv
```

Open `outputGeometry/calibration_lines_preview.jpg`, read the line ids, and group lines that are parallel in the real scene.

3. Update the calibration groups in `calibration_phase.py`:

```python
SELECTED_LINE_GROUPS = [
    [...],
    [...],
    [...],
]
```

If full intrinsic calibration is unstable, keep the principal point fixed near the image center by setting:

```python
ESTIMATE_PRINCIPAL_POINT = False
PRINCIPAL_POINT = (960.0, 540.0)
```

Use the correct image center if the frame resolution is not 1920x1080.

4. Run the main pipeline:

```bash
python main.py
```

The default pipeline runs:

```text
calibration -> slope -> viz -> trajectory
```

During the slope step, the GUI asks for:

- normal-direction segments: click pairs of points, then press `Enter`
- polyline A: click ordered points on the first track border, then press `Enter`
- polyline B: click ordered points on the second track border, then press `Enter`

## External resources

Ski jumping video samples:
https://drive.google.com/drive/folders/1NacD2FmqXp9dofT87Ru5cCQNJ2sDAhCa?usp=sharing

Roboflow annotated data:
https://app.roboflow.com/skijumpinglabeling
