from __future__ import annotations

from src.slope_geometry import extraction as _workflow
from src.slope_geometry.extraction import *  # noqa: F401,F403


_CONFIG_NAMES = (
    "FRAME_PATH",
    "K_JSON_PATH",
    "TRACK_WIDTH",
    "NUM_SAMPLES",
    "DISPLAY_SCALE",
    "MIN_NORMAL_SEGMENTS",
    "ANNOTATIONS_JSON_PATH",
    "OVERLAY_PATH",
    "CORRESPONDENCES_PATH",
    "RECONSTRUCTION_CSV_PATH",
    "RECONSTRUCTION_NPZ_PATH",
    "RECONSTRUCTION_3D_PATH",
    "DENSE_CURVE_SAMPLES",
    "ARC_LENGTH_REGULARIZATION_PX",
    "WINDOW_NAME",
)


def _sync_config() -> None:
    for name in _CONFIG_NAMES:
        setattr(_workflow, name, globals()[name])


def main() -> int:
    _sync_config()
    return _workflow.main()


if __name__ == "__main__":
    raise SystemExit(main())
