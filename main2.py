
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import calibration_phase
# import correspondences
import jumper_trajectory
import slope_extraction
import viz_slope


START_AT = "calibration"
STOP_AFTER = "trajectory"
SKIP_EXISTING = False
WITH_VIEWERS = False
PRINT_TRACEBACK = False


StepRunner = Callable[[bool], int | None]


@dataclass(frozen=True)
class PipelineStep:
    key: str
    title: str
    run: StepRunner
    outputs: tuple[Path, ...]


def run_calibration_phase(_with_viewers: bool) -> int | None:
    return calibration_phase.main()


# def run_correspondences(_with_viewers: bool) -> int | None:
#     return correspondences.main()


def run_slope_extraction(_with_viewers: bool) -> int | None:
    return slope_extraction.main()


def run_viz_slope(with_viewers: bool) -> int | None:
    if with_viewers:
        return viz_slope.main()

    reconstruction = viz_slope.load_reconstruction(viz_slope.RECONSTRUCTION_NPZ_PATH)
    annotations = viz_slope.load_annotations(viz_slope.ANNOTATIONS_JSON_PATH)
    _, bundle = viz_slope.prepare_view_reconstruction(reconstruction, annotations)
    if bundle is None:
        raise RuntimeError("viz_slope non ha prodotto il bundle di omografie della pista.")

    print(f"Omografie pista salvate: {viz_slope.HOMOGRAPHY_JSON_PATH}")
    print(f"Punti pista corretti salvati: {viz_slope.HOMOGRAPHY_POINTS_CSV_PATH}")
    return 0


def run_jumper_trajectory(with_viewers: bool) -> int | None:
    jumper_trajectory.OPEN_VIEWER = with_viewers
    return jumper_trajectory.main()

# Each PipelineStep describes one phase of the processing pipeline:
# key selects the phase, title is printed in the terminal, run executes it,
# and outputs lists the files that must exist after the phase succeeds.
PIPELINE_STEPS: tuple[PipelineStep, ...] = (
    # Phase: Calibration
    # Detects image lines, estimates vanishing points, and computes the camera matrix K.
    # This phase is responsible for saving the camera calibration used by all 3D steps.
    PipelineStep(
        key="calibration",
        title="Calibration phase",
        run=run_calibration_phase,
        outputs=(calibration_phase.OUTPUT_PATH,),
    ),
    # Phase: Correspondences
    # Automatically matches structural features on the right and left sides of the slope.
    # This phase estimates the horizontal vanishing point perpendicular to the slope lanes.
    # PipelineStep(
    #     key="correspondences",
    #     title="Automatic correspondences",
    #     run=run_correspondences,
    #     outputs=(
    #         correspondences.SUMMARY_JSON_PATH,
    #         correspondences.MATCHES_CSV_PATH,
    #     ),
    # ),
    # Phase: Slope extraction
    # Uses the camera calibration and user slope annotations to reconstruct the track borders in 3D.
    # This phase is responsible for saving annotation data and the slope reconstruction arrays.
    PipelineStep(
        key="slope",
        title="Slope extraction",
        run=run_slope_extraction,
        outputs=(
            slope_extraction.ANNOTATIONS_JSON_PATH,
            slope_extraction.RECONSTRUCTION_NPZ_PATH,
        ),
    ),
    # Phase: Slope visualization and geometry export
    # Loads the reconstructed slope, builds plane bases, and computes homographies for the slope planes.
    # This phase is responsible for saving corrected slope points and homographies used by trajectory lifting.
    PipelineStep(
        key="viz",
        title="Viz slope geometry",
        run=run_viz_slope,
        outputs=(
            viz_slope.HOMOGRAPHY_JSON_PATH,
            viz_slope.HOMOGRAPHY_POINTS_CSV_PATH,
        ),
    ),
    # Phase: Jumper trajectory
    # Stabilizes tracked jumper detections into the reference frame and lifts them onto the slope center plane.
    # This phase is responsible for saving reference-frame points and the final 3D jumper trajectory.
    PipelineStep(
        key="trajectory",
        title="Jumper trajectory",
        run=run_jumper_trajectory,
        outputs=(
            jumper_trajectory.REFERENCE_POINTS_CSV_PATH,
            jumper_trajectory.TRAJECTORY_NPZ_PATH,
        ),
    ),
)


def step_keys() -> list[str]:
    return [step.key for step in PIPELINE_STEPS]


def selected_steps(start_at: str, stop_after: str) -> tuple[PipelineStep, ...]:
    keys = step_keys()
    start = keys.index(start_at)
    stop = keys.index(stop_after)
    if start > stop:
        raise ValueError("START_AT deve precedere o coincidere con STOP_AFTER.")
    return PIPELINE_STEPS[start : stop + 1]


def missing_outputs(paths: Sequence[Path]) -> list[Path]:
    return [path for path in paths if not path.exists() or path.stat().st_size == 0]


def require_outputs(step: PipelineStep) -> None:
    missing = missing_outputs(step.outputs)
    if missing:
        formatted = ", ".join(str(path) for path in missing)
        raise RuntimeError(f"Output mancanti dopo {step.key}: {formatted}")


def run_step(step: PipelineStep, with_viewers: bool, skip_existing: bool) -> None:
    if skip_existing and not missing_outputs(step.outputs):
        print(f"\n== {step.title} ==")
        print("Output gia presenti, step saltato.")
        return

    print(f"\n== {step.title} ==")
    started = time.perf_counter()
    exit_code = step.run(with_viewers)
    if exit_code not in (None, 0):
        raise RuntimeError(f"{step.key} terminato con exit code {exit_code}.")
    require_outputs(step)
    elapsed = time.perf_counter() - started
    print(f"Step completato in {elapsed:.1f}s.")


def main() -> int:
    try:
        steps = selected_steps(START_AT, STOP_AFTER)
        print("Pipeline:")
        for step in steps:
            print(f"  - {step.key}")

        for step in steps:
            run_step(step, WITH_VIEWERS, SKIP_EXISTING)

    except Exception as error:
        print(f"\nPipeline interrotta: {error}")
        if PRINT_TRACEBACK:
            traceback.print_exc()
        return 1

    print("\nPipeline completata.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
