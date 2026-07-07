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


def extract_features(config: CalibrationConfig) -> list[tuple[int, int, int, int]]:
    frame = cv2.imread(str(config.frame_path))
    if frame is None:
        raise FileNotFoundError(
            f"Frame non trovato o non leggibile: {config.frame_path}. "
            "Esegui prima video_frame_picker.py per salvare il frame di lavoro.",
        )

    enhanced, edges = preprocess_for_lines(frame)
    lines = detect_lines(enhanced, edges, max_lines=config.max_lines)
    if len(lines) < 6:
        raise RuntimeError(f"Il detector ha trovato solo {len(lines)} linee utili; ne servono almeno 6.")

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
    print(f"Linee rilevate: {len(lines)}")
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

    return lines


def main() -> int:
    extract_features(build_config())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
