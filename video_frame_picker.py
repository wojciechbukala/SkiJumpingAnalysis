# Video frame selection workflow
# Step one resolves the main video path or fallback path
# Step two opens the video and reads fps total frame count and the starting frame
# Step three displays frames with overlay text for frame index time and controls
# Step four lets the user pause play step forward step backward print the selected frame or save it
# Step five optionally saves the selected frame to SELECTED_FRAME_OUTPUT
# Option VIDEO_PATH selects the primary video
# Option FALLBACK_VIDEO_PATHS lists backup locations
# Option DISPLAY_SCALE controls the preview size
# Option START_FRAME controls the initial frame
# Option PLAYBACK_DELAY_MS controls playback delay while not paused
# Option SAVE_SELECTED_FRAME controls automatic save after selection
# Formula time_seconds is frame_index divided by fps
# Formula display scaling uses width and height multiplied by DISPLAY_SCALE

from __future__ import annotations
from pathlib import Path
import cv2


VIDEO_PATH = Path("detectionOutputs/G_60out.mp4")
FALLBACK_VIDEO_PATHS = [
    Path("src/experiments/Detection_outputs_examples/G_30out.mp4"),
]

WINDOW_NAME = "G_60 frame picker"
DISPLAY_SCALE = 0.75
START_FRAME = 0
PLAYBACK_DELAY_MS = 30

SAVE_SELECTED_FRAME = True
SELECTED_FRAME_OUTPUT = Path("outputGeometry/selected_frame.jpg")


# Resolve the input video path
def resolve_video_path() -> Path:
    if VIDEO_PATH.exists():
        return VIDEO_PATH

    for path in FALLBACK_VIDEO_PATHS:
        if path.exists():
            return path

    candidates = [str(VIDEO_PATH), *[str(path) for path in FALLBACK_VIDEO_PATHS]]
    raise FileNotFoundError(f"Video G_30 non trovato. Percorsi provati: {candidates}")


# Format playback time
def format_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    remaining = seconds - minutes * 60
    return f"{minutes:02d}:{remaining:06.3f}"


# Read a requested video frame
def read_frame_at(cap: cv2.VideoCapture, frame_index: int) -> tuple[bool, object]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_index))
    return cap.read()


# Draw playback information
def draw_overlay(frame, frame_index: int, total_frames: int, fps: float, paused: bool):
    output = frame.copy()
    time_seconds = frame_index / fps if fps > 0 else 0.0
    status = "PAUSA" if paused else "PLAY"
    lines = [
        f"{status} | frame {frame_index}/{max(0, total_frames - 1)} | t={time_seconds:.3f}s ({format_time(time_seconds)})",
        "Spazio: pausa/play | frecce/A-D: indietro-avanti | Enter: seleziona e salva | S: salva | Q: esci",
    ]

    y = 34
    for line in lines:
        cv2.putText(output, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(output, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
        y += 34

    return output


# Scale a frame for display
def scaled_for_display(frame):
    if abs(DISPLAY_SCALE - 1.0) < 1e-9:
        return frame
    return cv2.resize(frame, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE, interpolation=cv2.INTER_AREA)


# Print the chosen frame
def print_selection(video_path: Path, frame_index: int, fps: float) -> None:
    time_seconds = frame_index / fps if fps > 0 else 0.0
    print("\nFrame selezionato")
    print(f"Video: {video_path}")
    print(f"Frame index: {frame_index}")
    print(f"Tempo: {time_seconds:.6f} s")
    print(f"Tempo leggibile: {format_time(time_seconds)}")


# Save the chosen frame
def save_selected_frame(frame, frame_index: int) -> None:
    SELECTED_FRAME_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(SELECTED_FRAME_OUTPUT), frame)
    print(f"Frame salvato: {SELECTED_FRAME_OUTPUT}")


# Run the frame selection workflow
def main() -> int:
    video_path = resolve_video_path()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire il video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        raise RuntimeError(f"FPS non valido: {fps}")
    if total_frames <= 0:
        raise RuntimeError(f"Numero frame non valido: {total_frames}")

    frame_index = max(0, min(START_FRAME, total_frames - 1))
    paused = False
    selected = False
    ret, frame = read_frame_at(cap, frame_index)
    if not ret:
        raise RuntimeError(f"Impossibile leggere il frame iniziale {frame_index}.")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    print(f"Video aperto: {video_path}")
    print(f"FPS: {fps:.6f}")
    print(f"Frame totali: {total_frames}")
    print(f"Premi Enter sul frame desiderato per salvarlo in: {SELECTED_FRAME_OUTPUT}")

    while True:
        cv2.imshow(WINDOW_NAME, scaled_for_display(draw_overlay(frame, frame_index, total_frames, fps, paused)))
        key = cv2.waitKeyEx(0 if paused else PLAYBACK_DELAY_MS)

        if key in (ord("q"), ord("Q"), 27):
            break

        if key in (13, 10):
            print_selection(video_path, frame_index, fps)
            selected = True
            break

        if key in (ord(" "),):
            paused = not paused
            continue

        if key in (ord("s"), ord("S")):
            save_selected_frame(frame, frame_index)
            continue

        step = 0
        if key in (ord("d"), ord("D"), 83, 2555904):
            step = 1
            paused = True
        elif key in (ord("a"), ord("A"), 81, 2424832):
            step = -1
            paused = True

        if paused and step != 0:
            frame_index = max(0, min(total_frames - 1, frame_index + step))
            ret, frame = read_frame_at(cap, frame_index)
            if not ret:
                print(f"Impossibile leggere il frame {frame_index}.")
            continue

        if not paused:
            next_frame_index = frame_index + 1
            if next_frame_index >= total_frames:
                paused = True
                continue
            ret, frame = cap.read()
            if not ret:
                paused = True
                continue
            frame_index = next_frame_index

    if SAVE_SELECTED_FRAME and selected:
        save_selected_frame(frame, frame_index)

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
