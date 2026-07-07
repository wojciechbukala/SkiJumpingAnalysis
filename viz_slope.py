from __future__ import annotations

from src.slope_geometry import visualization as _workflow
from src.slope_geometry.visualization import *  # noqa: F401,F403


_CONFIG_NAMES = (
    "RECONSTRUCTION_NPZ_PATH",
    "ANNOTATIONS_JSON_PATH",
    "HOMOGRAPHY_JSON_PATH",
    "HOMOGRAPHY_POINTS_CSV_PATH",
    "VIEWER_BACKEND",
    "WINDOW_TITLE",
    "USE_HOMOGRAPHY_CORRECTED_POINTS",
    "SHOW_CENTERLINE",
    "SHOW_RULINGS",
    "SHOW_NORMAL",
    "SHOW_POINTS",
    "SHOW_AXES",
    "SHOW_CAMERA",
    "SHOW_PLANE_BASES",
    "RULING_STRIDE",
    "NORMAL_SCALE",
    "CAMERA_FRUSTUM_SCALE",
    "CAMERA_AXIS_SCALE",
    "PLANE_BASIS_SCALE",
    "POINT_SIZE",
    "MATPLOTLIB_LINE_WIDTH",
)


def _sync_config() -> None:
    for name in _CONFIG_NAMES:
        setattr(_workflow, name, globals()[name])


def main() -> int:
    _sync_config()
    return _workflow.main()


if __name__ == "__main__":
    raise SystemExit(main())
