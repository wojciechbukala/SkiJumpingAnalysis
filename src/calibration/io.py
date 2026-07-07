from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def lines_metadata_path(lines_csv_path: Path) -> Path:
    return lines_csv_path.with_name(f"{lines_csv_path.stem}_metadata.json")


def load_lines_metadata(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Metadata linee non trovato: {path}")

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_lines_metadata(path: Path, frame_path: Path, line_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "frame_path": str(frame_path),
        "frame_mtime_ns": frame_path.stat().st_mtime_ns,
        "line_count": int(line_count),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def load_lines_csv(path: Path) -> list[tuple[int, int, int, int]]:
    if not path.exists():
        raise FileNotFoundError(f"Linee non trovate: {path}")

    lines = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"x1", "y1", "x2", "y2"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV linee incompleto, mancano colonne: {sorted(missing)}")

        for row in reader:
            lines.append(
                (
                    int(float(row["x1"])),
                    int(float(row["y1"])),
                    int(float(row["x2"])),
                    int(float(row["y2"])),
                ),
            )

    return lines


def save_lines_csv(path: Path, lines: list[tuple[int, int, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("id,x1,y1,x2,y2,length,angle_deg\n")
        for idx, (x1, y1, x2, y2) in enumerate(lines):
            dx = x2 - x1
            dy = y2 - y1
            length = float(np.hypot(dx, dy))
            angle = float(np.degrees(np.arctan2(dy, dx)) % 180.0)
            handle.write(f"{idx},{x1},{y1},{x2},{y2},{length:.3f},{angle:.3f}\n")
