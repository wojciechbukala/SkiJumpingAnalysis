from __future__ import annotations

import json

import cv2
import numpy as np

from src.calibration.config import CalibrationConfig
from src.calibration.io import lines_metadata_path, load_lines_csv, load_lines_metadata
from src.calibration.vanishing import (
    print_vanishing_points,
    vanishing_points_from_line_groups,
)
from src.track_geometry.calibration import (
    CalibrationError,
    estimate_k_from_orthogonal_vanishing_pairs,
    estimate_k_from_orthogonal_vanishing_pairs_and_equal_lengths,
)
from src.track_geometry.tool_line import (
    choose_equal_length_segments,
    choose_line_groups,
    draw_equal_length_segments_preview,
)


def estimate_from_line_groups(
    frame: np.ndarray,
    lines: list[tuple[int, int, int, int]],
    selected_groups: list[list[int]],
    equal_length_segments: list[tuple[tuple[float, float], tuple[float, float]]] | None,
    config: CalibrationConfig,
) -> tuple[np.ndarray, list[list[float]], list[list[list[float]]]]:
    vanishing_points = vanishing_points_from_line_groups(lines, selected_groups)
    print_vanishing_points(selected_groups, vanishing_points)

    pairs = [
        [vanishing_points[0], vanishing_points[1]],
        [vanishing_points[0], vanishing_points[2]],
        [vanishing_points[1], vanishing_points[2]],
    ]

    if config.estimate_principal_point and config.principal_point is None:
        if equal_length_segments is None:
            raise CalibrationError(
                "ESTIMATE_PRINCIPAL_POINT=True richiede due segmenti di uguale lunghezza reale "
                "oppure un altro vincolo metrico indipendente.",
            )
        normal_group_index = int(config.equal_length_plane_normal_group) - 1
        if normal_group_index < 0 or normal_group_index >= len(vanishing_points):
            raise ValueError("EQUAL_LENGTH_PLANE_NORMAL_GROUP deve essere 1, 2 o 3.")
        K = estimate_k_from_orthogonal_vanishing_pairs_and_equal_lengths(
            pairs,
            equal_length_segments,
            vanishing_points[normal_group_index],
            image_shape=frame.shape[:2],
        )
        return K, vanishing_points, pairs

    K = estimate_k_from_orthogonal_vanishing_pairs(
        pairs,
        image_shape=frame.shape[:2],
        principal_point=config.principal_point,
    )
    return K, vanishing_points, pairs


def run_calibration(config: CalibrationConfig) -> int:
    frame = cv2.imread(str(config.frame_path))
    if frame is None:
        raise FileNotFoundError(
            f"Frame non trovato o non leggibile: {config.frame_path}. "
            "Esegui prima video_frame_picker.py per salvare il frame di lavoro.",
        )

    try:
        lines = load_lines_csv(config.lines_csv_path)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Report linee non trovato: {config.lines_csv_path}. "
            "Esegui prima feature_extraction.py e poi inserisci SELECTED_LINE_GROUPS in calibration_phase.py.",
        ) from error
    metadata_path = lines_metadata_path(config.lines_csv_path)
    try:
        metadata = load_lines_metadata(metadata_path)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Metadata linee non trovato: {metadata_path}. "
            "Esegui di nuovo feature_extraction.py sul frame salvato dal picker.",
        ) from error
    if metadata.get("frame_path") != str(config.frame_path):
        raise RuntimeError(
            f"Le linee in {config.lines_csv_path} sono state estratte da {metadata.get('frame_path')}, "
            f"ma la calibrazione usa {config.frame_path}. Esegui di nuovo feature_extraction.py.",
        )
    if int(metadata.get("frame_mtime_ns", -1)) != config.frame_path.stat().st_mtime_ns:
        raise RuntimeError(
            f"Il frame {config.frame_path} e il report linee non sono sincronizzati. "
            "Esegui di nuovo feature_extraction.py.",
        )
    if config.lines_csv_path.stat().st_mtime < config.frame_path.stat().st_mtime:
        raise RuntimeError(
            f"Report linee piu vecchio del frame: {config.lines_csv_path}. "
            "Esegui di nuovo feature_extraction.py sul frame salvato dal picker.",
        )
    if len(lines) < 6:
        raise RuntimeError(f"Il report contiene solo {len(lines)} linee utili; ne servono almeno 6.")

    print(f"Frame usato: {config.frame_path}")
    print(f"Linee caricate: {len(lines)}")
    print(f"Report linee: {config.lines_csv_path}")

    if config.preview_only:
        if config.selected_line_groups is not None or config.selected_line_ids is not None:
            selected_groups = choose_line_groups(lines, config.selected_line_groups, config.selected_line_ids)
            vanishing_points = vanishing_points_from_line_groups(lines, selected_groups)
            print_vanishing_points(selected_groups, vanishing_points)
        print("Linee caricate. Imposta PREVIEW_ONLY = False e SELECTED_LINE_GROUPS = [[...], [...], [...]] per stimare K.")
        return 0

    if config.show_window:
        print(f"Preview linee numerate: {config.preview_path}")

    selected_groups = choose_line_groups(lines, config.selected_line_groups, config.selected_line_ids)
    equal_length_segments = None
    if config.estimate_principal_point and config.principal_point is None:
        equal_length_segments = choose_equal_length_segments(
            frame,
            config.equal_length_segments,
            draw_segments=config.draw_equal_length_segments,
            show_window=config.show_window,
        )
        equal_length_preview = draw_equal_length_segments_preview(frame, equal_length_segments)
        cv2.imwrite(str(config.equal_length_preview_path), equal_length_preview)
        print(f"Segmenti di uguale lunghezza reale: {equal_length_segments}")
        print(f"Normale del piano dei segmenti: gruppo {config.equal_length_plane_normal_group}")
        print(f"Preview segmenti uguali: {config.equal_length_preview_path}")

    try:
        K, vanishing_points, pairs = estimate_from_line_groups(
            frame,
            lines,
            selected_groups,
            equal_length_segments,
            config,
        )
    except CalibrationError as error:
        print(f"Calibrazione non riuscita: {error}")
        return 1

    payload = {
        "frame": str(config.frame_path),
        "selected_line_groups": selected_groups,
        "equal_length_segments": equal_length_segments,
        "equal_length_plane_normal_group": (
            config.equal_length_plane_normal_group if equal_length_segments is not None else None
        ),
        "vanishing_points": vanishing_points,
        "orthogonal_pairs": pairs,
        "K": K.tolist(),
    }

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    with config.output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    print("Estimated Intrinsic Matrix K:")
    print(K)
    print(f"Risultato salvato in: {config.output_path}")

    if config.show_window:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0
