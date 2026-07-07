from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_FRAME_PATH = Path("outputGeometry/selected_frame.jpg")
DEFAULT_PREVIEW_PATH = Path("outputGeometry/calibration_lines_preview.jpg")
DEFAULT_EDGES_PATH = Path("outputGeometry/calibration_edges.jpg")
DEFAULT_LINES_CSV_PATH = Path("outputGeometry/calibration_lines.csv")
DEFAULT_EQUAL_LENGTH_PREVIEW_PATH = Path("outputGeometry/calibration_equal_length_segments.jpg")
DEFAULT_OUTPUT_PATH = Path("outputGeometry/calibration_K.json")


@dataclass(frozen=True)
class CalibrationConfig:
    frame_path: Path
    max_lines: int
    selected_line_groups: list[list[int]] | None
    selected_line_ids: list[int] | None
    equal_length_segments: list[tuple[tuple[float, float], tuple[float, float]]] | None
    equal_length_plane_normal_group: int
    preview_path: Path
    edges_path: Path
    lines_csv_path: Path
    equal_length_preview_path: Path
    output_path: Path
    show_window: bool
    draw_equal_length_segments: bool
    preview_only: bool
    principal_point: tuple[float, float] | None
    estimate_principal_point: bool


def default_calibration_config() -> CalibrationConfig:
    return CalibrationConfig(
        frame_path=DEFAULT_FRAME_PATH,
        max_lines=120,
        selected_line_groups=[
            [82,44,50,9 , 13, 3, 22],
            [33, 81, 37, 42, 11, 1, 40, 7, 10],
            [70, 90, 55, 34,76],
        ],
        selected_line_ids=None,
        equal_length_segments=None,
        equal_length_plane_normal_group=3,
        preview_path=DEFAULT_PREVIEW_PATH,
        edges_path=DEFAULT_EDGES_PATH,
        lines_csv_path=DEFAULT_LINES_CSV_PATH,
        equal_length_preview_path=DEFAULT_EQUAL_LENGTH_PREVIEW_PATH,
        output_path=DEFAULT_OUTPUT_PATH,
        show_window=False,
        draw_equal_length_segments=True,
        preview_only=False,
        principal_point=None,
        estimate_principal_point=True,
    )
