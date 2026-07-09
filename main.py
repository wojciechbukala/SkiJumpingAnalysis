from __future__ import annotations

import jumper_trajectory

from src.pipeline.main2_steps import PIPELINE_STEPS
from src.pipeline.runner import run_pipeline


START_AT = "calibration"
STOP_AFTER = "trajectory"
SKIP_EXISTING = False
WITH_VIEWERS = False
PRINT_TRACEBACK = False
JUMPER_TRAJECTORY_CAMERA_ID = 1


def main() -> int:
    jumper_trajectory.SELECTED_CAMERA_ID = JUMPER_TRAJECTORY_CAMERA_ID
    return run_pipeline(
        PIPELINE_STEPS,
        START_AT,
        STOP_AFTER,
        WITH_VIEWERS,
        SKIP_EXISTING,
        PRINT_TRACEBACK,
    )


if __name__ == "__main__":
    raise SystemExit(main())
