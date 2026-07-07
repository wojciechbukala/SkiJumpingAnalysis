from __future__ import annotations

import calibration_phase
import jumper_trajectory
import slope_extraction
import viz_slope

from src.pipeline.runner import PipelineStep


def run_calibration_phase(_with_viewers: bool) -> int | None:
    return calibration_phase.main()


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


def run_jumper_trajectory(_with_viewers: bool) -> int | None:
    jumper_trajectory.OPEN_VIEWER = True
    jumper_trajectory.VIEWER_BACKEND = "matplotlib"
    return jumper_trajectory.main()


# Each step describes one phase of the processing pipeline.
PIPELINE_STEPS: tuple[PipelineStep, ...] = (
    PipelineStep(
        key="calibration",
        title="Calibration phase",
        run=run_calibration_phase,
        outputs=(calibration_phase.OUTPUT_PATH,),
    ),
    PipelineStep(
        key="slope",
        title="Slope extraction",
        run=run_slope_extraction,
        outputs=(
            slope_extraction.ANNOTATIONS_JSON_PATH,
            slope_extraction.RECONSTRUCTION_NPZ_PATH,
        ),
    ),
    PipelineStep(
        key="viz",
        title="Viz slope geometry",
        run=run_viz_slope,
        outputs=(
            viz_slope.HOMOGRAPHY_JSON_PATH,
            viz_slope.HOMOGRAPHY_POINTS_CSV_PATH,
        ),
    ),
    PipelineStep(
        key="trajectory",
        title="Jumper trajectory",
        run=run_jumper_trajectory,
        outputs=(
            jumper_trajectory.REFERENCE_POINTS_CSV_PATH,
            jumper_trajectory.TRAJECTORY_PNG_PATH,
        ),
    ),
)
