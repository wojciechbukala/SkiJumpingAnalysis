from __future__ import annotations

import cv2
import numpy as np

from calibration_phase import (
    EDGES_PATH as CALIBRATION_EDGES_PATH,
    EQUAL_LENGTH_PREVIEW_PATH as CALIBRATION_EQUAL_LENGTH_PREVIEW_PATH,
    FRAME_PATH as CALIBRATION_FRAME_PATH,
    LINES_CSV_PATH as CALIBRATION_LINES_CSV_PATH,
    MAX_LINES as CALIBRATION_MAX_LINES,
    OUTPUT_PATH as CALIBRATION_OUTPUT_PATH,
    PREVIEW_PATH as CALIBRATION_PREVIEW_PATH,
    SHOW_WINDOW as CALIBRATION_SHOW_WINDOW,
)
from src.calibration.config import CalibrationConfig
from src.calibration.io import lines_metadata_path, save_lines_csv, save_lines_metadata
from src.track_geometry.line_extraction import (
    build_search_edges,
    detect_lines,
    preprocess_for_lines,
)
from src.track_geometry.tool_line import draw_numbered_lines


FRAME_PATH = CALIBRATION_FRAME_PATH
MAX_LINES = CALIBRATION_MAX_LINES
PREVIEW_PATH = CALIBRATION_PREVIEW_PATH
EDGES_PATH = CALIBRATION_EDGES_PATH
LINES_CSV_PATH = CALIBRATION_LINES_CSV_PATH
EQUAL_LENGTH_PREVIEW_PATH = CALIBRATION_EQUAL_LENGTH_PREVIEW_PATH
OUTPUT_PATH = CALIBRATION_OUTPUT_PATH
SHOW_WINDOW = CALIBRATION_SHOW_WINDOW
MIN_REQUIRED_LINES = 6
MODE_AUTO = "auto"
MODE_MANUAL = "manual"
MANUAL_WINDOW_NAME = "Feature extraction - selezione manuale linee"


def build_config() -> CalibrationConfig:
    return CalibrationConfig(
        frame_path=FRAME_PATH,
        max_lines=MAX_LINES,
        selected_line_groups=None,
        selected_line_ids=None,
        equal_length_segments=None,
        equal_length_plane_normal_group=3,
        preview_path=PREVIEW_PATH,
        edges_path=EDGES_PATH,
        lines_csv_path=LINES_CSV_PATH,
        equal_length_preview_path=EQUAL_LENGTH_PREVIEW_PATH,
        output_path=OUTPUT_PATH,
        show_window=SHOW_WINDOW,
        draw_equal_length_segments=False,
        preview_only=True,
        principal_point=None,
        estimate_principal_point=False,
    )


def ask_extraction_mode() -> str:
    while True:
        raw = input("Modalita feature extraction [auto/manuale] (auto): ").strip().lower()
        try:
            return normalize_extraction_mode(raw)
        except ValueError:
            print("Risposta non valida: scrivi 'auto' oppure 'manuale'.")


def normalize_extraction_mode(raw: str) -> str:
    mode = raw.strip().lower()
    if mode in ("", "a", "auto", "automatico"):
        return MODE_AUTO
    if mode in ("m", "manual", "manuale"):
        return MODE_MANUAL
    raise ValueError(f"Modalita feature extraction non valida: {raw!r}.")


def display_scale_for_frame(frame: np.ndarray) -> float:
    height, width = frame.shape[:2]
    return min(1.0, 1400.0 / float(width), 900.0 / float(height))


def scaled_point(point: tuple[float, float], scale: float) -> tuple[int, int]:
    return int(round(point[0] * scale)), int(round(point[1] * scale))


def resize_for_display(image: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return image.copy()
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def draw_text_with_background(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.62
    thickness = 1
    size, baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    cv2.rectangle(
        image,
        (x - 5, y - size[1] - 6),
        (x + size[0] + 5, y + baseline + 5),
        (0, 0, 0),
        -1,
    )
    cv2.putText(image, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_numbered_lines_scaled(
    frame: np.ndarray,
    lines: list[tuple[int, int, int, int]],
    scale: float,
) -> np.ndarray:
    return resize_for_display(draw_numbered_lines(frame, lines), scale)


def draw_manual_line_editor_frame(
    frame: np.ndarray,
    lines: list[tuple[int, int, int, int]],
    pending_start: tuple[float, float] | None,
    mouse_position: tuple[float, float] | None,
    status_message: str,
    scale: float,
) -> np.ndarray:
    preview = draw_numbered_lines_scaled(frame, lines, scale)
    if pending_start is not None:
        start = scaled_point(pending_start, scale)
        cv2.circle(preview, start, 6, (0, 255, 255), -1, cv2.LINE_AA)
        if mouse_position is not None:
            end = scaled_point(mouse_position, scale)
            cv2.line(preview, start, end, (0, 255, 255), 2, cv2.LINE_AA)

    draw_text_with_background(preview, "Manuale: click sinistro = primo/secondo estremo linea", (24, 36))
    draw_text_with_background(
        preview,
        "Enter/spazio/click destro conferma | u undo | c annulla punto | r reset | q/Esc annulla",
        (24, 66),
    )
    draw_text_with_background(preview, f"Linee selezionate: {len(lines)}", (24, 96))
    if status_message:
        draw_text_with_background(preview, status_message, (24, 126))
    return preview


def manual_line_from_points(
    first: tuple[float, float],
    second: tuple[float, float],
    frame_shape: tuple[int, ...],
) -> tuple[int, int, int, int] | None:
    if float(np.hypot(second[0] - first[0], second[1] - first[1])) <= 1.0:
        return None

    height, width = frame_shape[:2]
    x1 = int(round(max(0.0, min(float(width - 1), first[0]))))
    y1 = int(round(max(0.0, min(float(height - 1), first[1]))))
    x2 = int(round(max(0.0, min(float(width - 1), second[0]))))
    y2 = int(round(max(0.0, min(float(height - 1), second[1]))))
    return x1, y1, x2, y2


def request_manual_lines(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    scale = display_scale_for_frame(frame)
    state: dict[str, object] = {
        "lines": [],
        "pending_start": None,
        "mouse_position": None,
        "done": False,
        "cancelled": False,
        "status": "Seleziona almeno 6 linee.",
    }

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        image_point = (float(x) / scale, float(y) / scale)
        if event == cv2.EVENT_MOUSEMOVE:
            state["mouse_position"] = image_point
            return

        lines = state["lines"]
        if not isinstance(lines, list):
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            pending_start = state["pending_start"]
            if pending_start is None:
                state["pending_start"] = image_point
                state["status"] = "Secondo click per chiudere la linea."
                return
            if not isinstance(pending_start, tuple):
                return

            line = manual_line_from_points(pending_start, image_point, frame.shape)
            state["pending_start"] = None
            if line is None:
                state["status"] = "Linea troppo corta: scegli due punti distinti."
                return
            lines.append(line)
            state["status"] = f"Linea aggiunta: {len(lines)}."
        elif event == cv2.EVENT_RBUTTONDOWN:
            if state["pending_start"] is not None:
                state["pending_start"] = None
                state["status"] = "Punto iniziale annullato."
            elif len(lines) >= MIN_REQUIRED_LINES:
                state["done"] = True
            else:
                state["status"] = f"Servono almeno {MIN_REQUIRED_LINES} linee prima di confermare."

    try:
        cv2.namedWindow(MANUAL_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(MANUAL_WINDOW_NAME, on_mouse)

        while True:
            lines = state["lines"]
            if not isinstance(lines, list):
                lines = []
            pending_start = state["pending_start"]
            mouse_position = state["mouse_position"]
            status = state["status"] if isinstance(state["status"], str) else ""
            preview = draw_manual_line_editor_frame(
                frame,
                lines,
                pending_start if isinstance(pending_start, tuple) else None,
                mouse_position if isinstance(mouse_position, tuple) else None,
                status,
                scale,
            )
            cv2.imshow(MANUAL_WINDOW_NAME, preview)

            try:
                if cv2.getWindowProperty(MANUAL_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    state["cancelled"] = True
                    break
            except cv2.error:
                state["cancelled"] = True
                break

            key = cv2.waitKey(20) & 0xFF
            lines = state["lines"]
            if not isinstance(lines, list):
                lines = []

            if key in (13, 10, 32):
                if state["pending_start"] is not None:
                    state["status"] = "Completa o annulla la linea corrente prima di confermare."
                elif len(lines) >= MIN_REQUIRED_LINES:
                    state["done"] = True
                else:
                    state["status"] = f"Servono almeno {MIN_REQUIRED_LINES} linee prima di confermare."
            elif key == ord("u"):
                if state["pending_start"] is not None:
                    state["pending_start"] = None
                    state["status"] = "Punto iniziale annullato."
                elif lines:
                    lines.pop()
                    state["status"] = f"Ultima linea rimossa: {len(lines)} rimaste."
            elif key == ord("c"):
                state["pending_start"] = None
                state["status"] = "Punto iniziale annullato."
            elif key == ord("r"):
                lines.clear()
                state["pending_start"] = None
                state["status"] = "Linee resettate."
            elif key in (ord("q"), 27):
                state["cancelled"] = True

            if state["done"] or state["cancelled"]:
                break

        cv2.destroyWindow(MANUAL_WINDOW_NAME)
    except cv2.error as error:
        print(f"OpenCV GUI non disponibile per la selezione manuale delle linee: {error}")
        return []

    if state["cancelled"]:
        print("Selezione manuale linee annullata.")
        return []

    lines = state["lines"]
    if not isinstance(lines, list):
        return []
    return lines


def collect_manual_lines(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    print("Selezione manuale: clicca due estremi per ogni linea, poi Enter/spazio per confermare.")
    lines = request_manual_lines(frame)
    if len(lines) < MIN_REQUIRED_LINES:
        raise RuntimeError(
            f"La selezione manuale contiene solo {len(lines)} linee; ne servono almeno {MIN_REQUIRED_LINES}.",
        )
    return lines


def save_feature_outputs(
    config: CalibrationConfig,
    frame: np.ndarray,
    edges: np.ndarray,
    lines: list[tuple[int, int, int, int]],
    mode: str,
) -> None:
    config.preview_path.parent.mkdir(parents=True, exist_ok=True)
    preview = draw_numbered_lines(frame, lines)
    search_edges_preview = np.zeros_like(edges)
    search_y0, search_edges = build_search_edges(edges)
    search_edges_preview[search_y0:, :] = search_edges

    cv2.imwrite(str(config.preview_path), preview)
    cv2.imwrite(str(config.edges_path), search_edges_preview)
    save_lines_csv(config.lines_csv_path, lines)
    metadata_path = lines_metadata_path(config.lines_csv_path)
    save_lines_metadata(metadata_path, config.frame_path, len(lines))

    print(f"Frame usato: {config.frame_path}")
    print(f"Modalita: {'manuale' if mode == MODE_MANUAL else 'auto'}")
    print(f"Linee salvate: {len(lines)}")
    print(f"Preview linee numerate: {config.preview_path}")
    print(f"Edges filtrati: {config.edges_path}")
    print(f"Report linee: {config.lines_csv_path}")
    print(f"Metadata linee: {metadata_path}")
    print("Apri calibration_lines_preview.jpg e inserisci i gruppi in calibration_phase.py:")
    print("SELECTED_LINE_GROUPS = [[...], [...], [...]]")

    if config.show_window:
        try:
            cv2.imshow("Feature extraction - linee numerate", preview)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            print("OpenCV GUI non disponibile: usa il file preview salvato.")


def extract_features(config: CalibrationConfig, mode: str = MODE_AUTO) -> list[tuple[int, int, int, int]]:
    mode = normalize_extraction_mode(mode)
    frame = cv2.imread(str(config.frame_path))
    if frame is None:
        raise FileNotFoundError(
            f"Frame non trovato o non leggibile: {config.frame_path}. "
            "Esegui prima video_frame_picker.py per salvare il frame di lavoro.",
        )

    enhanced, edges = preprocess_for_lines(frame)
    if mode == MODE_AUTO:
        lines = detect_lines(enhanced, edges, max_lines=config.max_lines)
        if len(lines) < MIN_REQUIRED_LINES:
            raise RuntimeError(
                f"Il detector ha trovato solo {len(lines)} linee utili; ne servono almeno {MIN_REQUIRED_LINES}.",
            )
    elif mode == MODE_MANUAL:
        lines = collect_manual_lines(frame)
    else:
        raise ValueError(f"Modalita feature extraction non valida: {mode!r}.")

    save_feature_outputs(config, frame, edges, lines, mode)
    return lines


def main() -> int:
    extract_features(build_config(), ask_extraction_mode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
