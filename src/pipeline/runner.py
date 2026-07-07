from __future__ import annotations

import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path


StepRunner = Callable[[bool], int | None]


@dataclass(frozen=True)
class PipelineStep:
    key: str
    title: str
    run: StepRunner
    outputs: tuple[Path, ...]


def step_keys(steps: Sequence[PipelineStep]) -> list[str]:
    return [step.key for step in steps]


def selected_steps(
    steps: Sequence[PipelineStep],
    start_at: str,
    stop_after: str,
) -> tuple[PipelineStep, ...]:
    keys = step_keys(steps)
    start = keys.index(start_at)
    stop = keys.index(stop_after)
    if start > stop:
        raise ValueError("START_AT deve precedere o coincidere con STOP_AFTER.")
    return tuple(steps[start : stop + 1])


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
    exit_code = step.run(with_viewers)
    if exit_code not in (None, 0):
        raise RuntimeError(f"{step.key} terminato con exit code {exit_code}.")
    require_outputs(step)


def run_pipeline(
    steps: Sequence[PipelineStep],
    start_at: str,
    stop_after: str,
    with_viewers: bool,
    skip_existing: bool,
    print_traceback: bool,
) -> int:
    try:
        selected = selected_steps(steps, start_at, stop_after)
        print("Pipeline:")
        for step in selected:
            print(f"  - {step.key}")

        for step in selected:
            run_step(step, with_viewers, skip_existing)

    except Exception as error:
        print(f"\nPipeline interrotta: {error}")
        if print_traceback:
            traceback.print_exc()
        return 1

    print("\nPipeline completata.")
    return 0
