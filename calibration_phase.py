from __future__ import annotations

from pathlib import Path

from video_frame_picker import SELECTED_FRAME_OUTPUT

from src.calibration.config import CalibrationConfig
from src.calibration.io import save_lines_csv
from src.calibration.vanishing import (
    line_from_segment,
    print_vanishing_points,
    vanishing_point_from_line_group,
    vanishing_points_from_line_groups,
)
from src.calibration.workflow import estimate_from_line_groups as _estimate_from_line_groups
from src.calibration.workflow import run_calibration


FRAME_PATH = SELECTED_FRAME_OUTPUT
MAX_LINES = 120
SELECTED_LINE_GROUPS = [
    [15,16,13,9,11,10,12,14],
    [5,7,8,17,6],
    [0,1,3,4],
]
EQUAL_LENGTH_SEGMENTS: list[tuple[tuple[float, float], tuple[float, float]]] | None = None
EQUAL_LENGTH_PLANE_NORMAL_GROUP = 3
SELECTED_LINE_IDS: list[int] | None = None
PREVIEW_PATH = Path("outputGeometry/calibration_lines_preview.jpg")
EDGES_PATH = Path("outputGeometry/calibration_edges.jpg")
LINES_CSV_PATH = Path("outputGeometry/calibration_lines.csv")
EQUAL_LENGTH_PREVIEW_PATH = Path("outputGeometry/calibration_equal_length_segments.jpg")
OUTPUT_PATH = Path("outputGeometry/calibration_K.json")
SHOW_WINDOW = False
DRAW_EQUAL_LENGTH_SEGMENTS = True
PREVIEW_ONLY = False
PRINCIPAL_POINT: tuple[float, float] | None = None
ESTIMATE_PRINCIPAL_POINT = True


def build_config() -> CalibrationConfig:
    return CalibrationConfig(
        frame_path=FRAME_PATH,
        max_lines=MAX_LINES,
        selected_line_groups=SELECTED_LINE_GROUPS,
        selected_line_ids=SELECTED_LINE_IDS,
        equal_length_segments=EQUAL_LENGTH_SEGMENTS,
        equal_length_plane_normal_group=EQUAL_LENGTH_PLANE_NORMAL_GROUP,
        preview_path=PREVIEW_PATH,
        edges_path=EDGES_PATH,
        lines_csv_path=LINES_CSV_PATH,
        equal_length_preview_path=EQUAL_LENGTH_PREVIEW_PATH,
        output_path=OUTPUT_PATH,
        show_window=SHOW_WINDOW,
        draw_equal_length_segments=DRAW_EQUAL_LENGTH_SEGMENTS,
        preview_only=PREVIEW_ONLY,
        principal_point=PRINCIPAL_POINT,
        estimate_principal_point=ESTIMATE_PRINCIPAL_POINT,
    )


def estimate_from_line_groups(frame, lines, selected_groups, equal_length_segments):
    return _estimate_from_line_groups(
        frame,
        lines,
        selected_groups,
        equal_length_segments,
        build_config(),
    )


def main() -> int:
    return run_calibration(build_config())


if __name__ == "__main__":
    raise SystemExit(main())
