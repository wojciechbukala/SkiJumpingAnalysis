from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

from src.jumperTrajectory.config import TrajectoryConfig
from viz_slope import (
    BACKGROUND_COLOR,
    CAMERA_COLOR,
    CAMERA_X_COLOR,
    CAMERA_Y_COLOR,
    CAMERA_Z_COLOR,
    CENTER_COLOR,
    LEFT_COLOR,
    RIGHT_COLOR,
    camera_axis_points,
    camera_frustum_points,
    set_axes_equal,
)


NON_INTERACTIVE_MATPLOTLIB_BACKENDS = {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}


def draw_camera_matplotlib(ax, K: np.ndarray, image_size: tuple[int, int], scene_points: np.ndarray) -> np.ndarray:
    corners = camera_frustum_points(K, image_size, scene_points)
    axis_points = camera_axis_points(scene_points)
    origin = np.zeros(3, dtype=float)
    for corner in corners:
        segment = np.vstack([origin, corner])
        ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color=CAMERA_COLOR, linewidth=1.0)
    for first, second in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        segment = np.vstack([corners[first], corners[second]])
        ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color=CAMERA_COLOR, linewidth=1.0)
    ax.scatter([0.0], [0.0], [0.0], color=CAMERA_COLOR, s=30, label="camera")
    ax.quiver(0.0, 0.0, 0.0, axis_points[1, 0], axis_points[1, 1], axis_points[1, 2], color=CAMERA_X_COLOR)
    ax.quiver(0.0, 0.0, 0.0, axis_points[2, 0], axis_points[2, 1], axis_points[2, 2], color=CAMERA_Y_COLOR)
    ax.quiver(0.0, 0.0, 0.0, axis_points[3, 0], axis_points[3, 1], axis_points[3, 2], color=CAMERA_Z_COLOR)
    return np.vstack([origin, corners, axis_points])


def draw_scene_matplotlib(
    ax,
    slope: dict,
    trajectory_3d: np.ndarray,
    frames: np.ndarray,
    K: np.ndarray,
    image_size: tuple[int, int],
    config: TrajectoryConfig,
) -> None:
    left = slope["left_3d"]
    right = slope["right_3d"]
    center = slope["center_3d"]
    all_points = np.vstack([left, right, center, trajectory_3d])

    ax.plot(left[:, 0], left[:, 1], left[:, 2], color=LEFT_COLOR, linewidth=config.line_width, label="corsia A")
    ax.plot(right[:, 0], right[:, 1], right[:, 2], color=RIGHT_COLOR, linewidth=config.line_width, label="corsia B")
    ax.plot(center[:, 0], center[:, 1], center[:, 2], "--", color=CENTER_COLOR, linewidth=1.5, label="linea centrale")

    stride = max(1, len(left) // 25)
    for idx in range(0, len(left), stride):
        segment = np.vstack([left[idx], right[idx]])
        ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color=config.ruling_color, linewidth=0.8, alpha=0.7)

    ax.plot(
        trajectory_3d[:, 0],
        trajectory_3d[:, 1],
        trajectory_3d[:, 2],
        color=config.trajectory_color,
        linewidth=2.5,
        label="traiettoria sciatore",
    )
    values = np.linspace(0.0, 1.0, len(trajectory_3d))
    scatter = ax.scatter(
        trajectory_3d[:, 0],
        trajectory_3d[:, 1],
        trajectory_3d[:, 2],
        c=values,
        cmap="plasma",
        s=config.point_size,
        label="frame",
    )
    if len(frames) > 0:
        ax.text(*trajectory_3d[0], f"{int(frames[0])}", color=config.trajectory_color)
        ax.text(*trajectory_3d[-1], f"{int(frames[-1])}", color=config.trajectory_color)

    camera_points = draw_camera_matplotlib(ax, K, image_size, all_points)
    all_points = np.vstack([all_points, camera_points])
    ax.set_xlabel("X camera")
    ax.set_ylabel("Y camera")
    ax.set_zlabel("Z camera")
    ax.set_title(f"Jumper trajectory on G_30, ref frame {config.reference_frame}")
    ax.legend(loc="upper right")
    set_axes_equal(ax, all_points)
    ax.view_init(elev=22, azim=-62)
    try:
        ax.figure.colorbar(scatter, ax=ax, shrink=0.65, pad=0.08, label="ordine temporale")
    except Exception:
        pass


def plot_static_3d(
    path: Path,
    slope: dict,
    trajectory_3d: np.ndarray,
    frames: np.ndarray,
    K: np.ndarray,
    image_size: tuple[int, int],
    config: TrajectoryConfig,
) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    path.parent.mkdir(parents=True, exist_ok=True)
    fig = Figure(figsize=(11, 8))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection="3d")
    draw_scene_matplotlib(ax, slope, trajectory_3d, frames, K, image_size, config)
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def is_non_interactive_matplotlib_backend(backend: str) -> bool:
    name = backend.rsplit(".", maxsplit=1)[-1].lower()
    return name in NON_INTERACTIVE_MATPLOTLIB_BACKENDS or name.endswith("agg")


def ensure_interactive_matplotlib_backend() -> None:
    import matplotlib

    current_backend = matplotlib.get_backend()
    if not is_non_interactive_matplotlib_backend(current_backend):
        return

    candidates = ["MacOSX", "QtAgg", "TkAgg"] if sys.platform == "darwin" else ["QtAgg", "TkAgg"]
    errors = []
    for candidate in candidates:
        try:
            if "matplotlib.pyplot" in sys.modules:
                import matplotlib.pyplot as plt

                plt.switch_backend(candidate)
            else:
                matplotlib.use(candidate, force=True)
            if not is_non_interactive_matplotlib_backend(matplotlib.get_backend()):
                return
        except Exception as error:
            errors.append(f"{candidate}: {error}")

    details = "; ".join(errors)
    raise RuntimeError(
        "Matplotlib sta usando un backend non interattivo "
        f"({current_backend}) e non e' stato possibile attivare un backend GUI. "
        f"Tentativi: {details}",
    )


def show_matplotlib_viewer(
    slope: dict,
    trajectory_3d: np.ndarray,
    frames: np.ndarray,
    K: np.ndarray,
    image_size: tuple[int, int],
    config: TrajectoryConfig,
) -> None:
    ensure_interactive_matplotlib_backend()
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    draw_scene_matplotlib(ax, slope, trajectory_3d, frames, K, image_size, config)
    fig.tight_layout()
    print(f"Viewer Matplotlib aperto con backend {plt.get_backend()}. Usa mouse e toolbar per ruotare, zoomare e spostarti.")
    plt.show()


def open3d_line_set(o3d, points: np.ndarray, lines: list[list[int]], colors: list[tuple[float, float, float]]):
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    line_set.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=float))
    return line_set


def camera_open3d_geometry(o3d, K: np.ndarray, image_size: tuple[int, int], scene_points: np.ndarray):
    corners = camera_frustum_points(K, image_size, scene_points)
    axis_points = camera_axis_points(scene_points)
    points = np.vstack([np.zeros((1, 3), dtype=float), corners, axis_points[1:]])
    lines = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 1],
        [0, 5],
        [0, 6],
        [0, 7],
    ]
    colors = [CAMERA_COLOR] * 8 + [CAMERA_X_COLOR, CAMERA_Y_COLOR, CAMERA_Z_COLOR]
    return open3d_line_set(o3d, points, lines, colors)


def show_open3d_viewer(
    slope: dict,
    trajectory_3d: np.ndarray,
    K: np.ndarray,
    image_size: tuple[int, int],
    config: TrajectoryConfig,
) -> None:
    try:
        import open3d as o3d
    except ImportError as error:
        raise RuntimeError("Open3D non installato: usa venv/bin/python -m pip install open3d") from error

    left = slope["left_3d"]
    right = slope["right_3d"]
    center = slope["center_3d"]
    points_blocks = [left, right, center, trajectory_3d]
    count_left = len(left)
    count_center = len(center)
    count_traj = len(trajectory_3d)
    offset_right = count_left
    offset_center = count_left + len(right)
    offset_traj = offset_center + count_center

    lines = []
    colors = []
    lines.extend([[idx, idx + 1] for idx in range(count_left - 1)])
    colors.extend([LEFT_COLOR] * (count_left - 1))
    lines.extend([[offset_right + idx, offset_right + idx + 1] for idx in range(len(right) - 1)])
    colors.extend([RIGHT_COLOR] * (len(right) - 1))
    lines.extend([[offset_center + idx, offset_center + idx + 1] for idx in range(count_center - 1)])
    colors.extend([CENTER_COLOR] * (count_center - 1))
    lines.extend([[offset_traj + idx, offset_traj + idx + 1] for idx in range(count_traj - 1)])
    colors.extend([config.trajectory_color] * max(0, count_traj - 1))

    stride = max(1, count_left // 25)
    for idx in range(0, count_left, stride):
        lines.append([idx, offset_right + idx])
        colors.append(config.ruling_color)

    scene_points = np.vstack(points_blocks)
    geometries = [open3d_line_set(o3d, scene_points, lines, colors)]
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(trajectory_3d)
    point_colors = np.tile(np.asarray(config.trajectory_color, dtype=float), (len(trajectory_3d), 1))
    point_cloud.colors = o3d.utility.Vector3dVector(point_colors)
    geometries.append(point_cloud)
    geometries.append(camera_open3d_geometry(o3d, K, image_size, scene_points))

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(window_name="Jumper trajectory 3D", width=1200, height=850)
    for geometry in geometries:
        visualizer.add_geometry(geometry)

    options = visualizer.get_render_option()
    options.background_color = np.asarray(BACKGROUND_COLOR, dtype=float)
    options.point_size = config.point_size
    options.line_width = 2.0
    print("Viewer Open3D aperto. Mouse sinistro ruota, rotella zoom, mouse destro o shift+sinistro sposta.")
    visualizer.run()
    visualizer.destroy_window()


def pyvista_polyline(pv, points: np.ndarray):
    polyline = pv.PolyData(np.asarray(points, dtype=float))
    if len(points) >= 2:
        polyline.lines = np.hstack([[len(points)], np.arange(len(points), dtype=np.int64)])
    return polyline


def pyvista_line_segments(pv, points: np.ndarray, lines: list[list[int]]):
    line_set = pv.PolyData(np.asarray(points, dtype=float))
    if lines:
        cells = np.asarray([[2, int(first), int(second)] for first, second in lines], dtype=np.int64)
        line_set.lines = cells.ravel()
    return line_set


def show_pyvista_viewer(
    slope: dict,
    trajectory_3d: np.ndarray,
    frames: np.ndarray,
    K: np.ndarray,
    image_size: tuple[int, int],
    config: TrajectoryConfig,
) -> None:
    try:
        import pyvista as pv
    except ImportError as error:
        raise RuntimeError("PyVista non installato: usa .venv/bin/python -m pip install pyvista") from error

    left = np.asarray(slope["left_3d"], dtype=float)
    right = np.asarray(slope["right_3d"], dtype=float)
    center = np.asarray(slope["center_3d"], dtype=float)
    trajectory_3d = np.asarray(trajectory_3d, dtype=float)
    scene_points = np.vstack([left, right, center, trajectory_3d])

    plotter = pv.Plotter(window_size=(1280, 860), title="Jumper trajectory 3D")
    plotter.set_background(BACKGROUND_COLOR)

    plotter.add_mesh(pyvista_polyline(pv, left), color=LEFT_COLOR, line_width=config.line_width, label="corsia A")
    plotter.add_mesh(pyvista_polyline(pv, right), color=RIGHT_COLOR, line_width=config.line_width, label="corsia B")
    plotter.add_mesh(pyvista_polyline(pv, center), color=CENTER_COLOR, line_width=1.5, label="linea centrale")

    stride = max(1, len(left) // 25)
    ruling_points = np.vstack([left, right])
    ruling_lines = [[idx, len(left) + idx] for idx in range(0, len(left), stride)]
    plotter.add_mesh(pyvista_line_segments(pv, ruling_points, ruling_lines), color=config.ruling_color, line_width=1.0)

    trajectory_line = pyvista_polyline(pv, trajectory_3d)
    plotter.add_mesh(trajectory_line, color=config.trajectory_color, line_width=4.0, label="traiettoria sciatore")
    plotter.add_points(
        trajectory_3d,
        color=config.trajectory_color,
        point_size=config.point_size,
        render_points_as_spheres=True,
    )
    if len(frames) > 0:
        label_points = np.vstack([trajectory_3d[0], trajectory_3d[-1]])
        labels = [str(int(frames[0])), str(int(frames[-1]))]
        plotter.add_point_labels(label_points, labels, font_size=14, text_color=config.trajectory_color, point_size=0)

    corners = camera_frustum_points(K, image_size, scene_points)
    axis_points = camera_axis_points(scene_points)
    camera_points = np.vstack([np.zeros((1, 3), dtype=float), corners, axis_points[1:]])
    camera_lines = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 1],
    ]
    plotter.add_mesh(pyvista_line_segments(pv, camera_points, camera_lines), color=CAMERA_COLOR, line_width=1.5)
    axis_lines = [[[0, 5]], [[0, 6]], [[0, 7]]]
    for lines, color in zip(axis_lines, [CAMERA_X_COLOR, CAMERA_Y_COLOR, CAMERA_Z_COLOR]):
        plotter.add_mesh(pyvista_line_segments(pv, camera_points, lines), color=color, line_width=3.0)
    plotter.add_points(np.zeros((1, 3), dtype=float), color=CAMERA_COLOR, point_size=10.0, render_points_as_spheres=True)

    plotter.add_axes()
    plotter.show_grid()
    plotter.add_legend()
    print("Viewer PyVista aperto. Usa il mouse per ruotare, zoomare e spostarti.")
    plotter.show(interactive=True)


def maybe_show_viewer(
    slope: dict,
    trajectory_3d: np.ndarray,
    frames: np.ndarray,
    K: np.ndarray,
    image_size: tuple[int, int],
    config: TrajectoryConfig,
) -> None:
    if not config.open_viewer:
        return

    backend = config.viewer_backend.lower().strip()
    if backend not in {"auto", "pyvista", "open3d", "matplotlib"}:
        raise ValueError("VIEWER_BACKEND deve essere 'auto', 'pyvista', 'open3d' o 'matplotlib'.")

    if backend in {"auto", "matplotlib"}:
        show_matplotlib_viewer(slope, trajectory_3d, frames, K, image_size, config)
        return

    if backend == "pyvista":
        show_pyvista_viewer(slope, trajectory_3d, frames, K, image_size, config)
        return

    if backend == "open3d":
        show_open3d_viewer(slope, trajectory_3d, K, image_size, config)
        return
