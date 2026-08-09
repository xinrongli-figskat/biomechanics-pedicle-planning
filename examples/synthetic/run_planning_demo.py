"""Run a fully synthetic entry-point scoring and optimization demonstration.

The generated volume and sphere are abstract numerical test objects. They do
not represent human anatomy, a clinical scan, or patient-derived geometry.
No input files are read by this script.
"""

from __future__ import annotations

import io
import os
from contextlib import redirect_stdout
from pathlib import Path
from time import perf_counter

import matplotlib
import numpy as np
import pyvista as pv
import SimpleITK as sitk

from pedicle_planning.ai.geometry import generate_search_sphere
from pedicle_planning.ai.optimizer import EntryPointOptimizer
from pedicle_planning.ai.scorer import EntryPointScorer
from pedicle_planning.ai.utils import clip_hu, normalize_hu


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

VOLUME_SHAPE = (48, 48, 48)
CENTER = np.array((24.0, 24.0, 24.0))
SURFACE_RADIUS = 16.0


class RecordingScorer:
    """Delegate scoring while recording evaluation points and values."""

    def __init__(self, scorer: EntryPointScorer) -> None:
        self._scorer = scorer
        self.points: list[tuple[float, float, float]] = []
        self.scores: list[float] = []

    def score(self, point: tuple[float, float, float]) -> float:
        normalized_point = tuple(float(value) for value in point)
        value = float(self._scorer.score(normalized_point))
        self.points.append(normalized_point)
        self.scores.append(value)
        return value


def create_synthetic_ct() -> tuple[sitk.Image, np.ndarray, np.ndarray]:
    """Create an abstract three-region HU volume entirely in memory."""
    z, y, x = np.indices(VOLUME_SHAPE, dtype=float)
    radius = np.sqrt(
        (x - CENTER[0]) ** 2
        + (y - CENTER[1]) ** 2
        + (z - CENTER[2]) ** 2
    )

    volume = np.full(VOLUME_SHAPE, -900, dtype=np.int16)
    volume[radius <= 14.0] = 250
    volume[(radius > 14.0) & (radius <= SURFACE_RADIUS)] = 1100

    image = sitk.GetImageFromArray(volume)
    image.SetOrigin((0.0, 0.0, 0.0))
    image.SetSpacing((1.0, 1.0, 1.0))
    image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

    clipped = clip_hu(volume)
    normalized = normalize_hu(clipped)
    return image, clipped, normalized


def create_synthetic_surface() -> pv.PolyData:
    """Create a generic sphere that is not intended to model anatomy."""
    return pv.Sphere(
        radius=SURFACE_RADIUS,
        center=tuple(CENTER),
        theta_resolution=48,
        phi_resolution=48,
    )


def save_planning_visualization(
    surface: pv.PolyData,
    candidates: list[tuple[float, float, float]],
    initial_point: tuple[float, float, float],
    optimized_point: tuple[float, float, float],
    output_path: Path,
) -> None:
    """Render a static view of the synthetic surface and search points."""
    points = np.asarray(surface.points)
    candidate_array = np.asarray(candidates)
    initial = np.asarray(initial_point)
    optimized = np.asarray(optimized_point)

    figure = plt.figure(figsize=(10, 8), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    stride = max(1, len(points) // 1200)
    sampled_surface = points[::stride]
    axis.scatter(
        sampled_surface[:, 0],
        sampled_surface[:, 1],
        sampled_surface[:, 2],
        s=3,
        alpha=0.12,
        color="#718096",
        label="Synthetic sphere surface",
    )
    axis.scatter(
        candidate_array[:, 0],
        candidate_array[:, 1],
        candidate_array[:, 2],
        s=16,
        alpha=0.55,
        color="#3182ce",
        label="Generated candidates",
    )
    axis.scatter(*initial, s=110, color="#dd6b20", label="Initial point")
    axis.scatter(*optimized, s=130, color="#c53030", marker="*", label="Optimized point")
    axis.plot(
        [initial[0], optimized[0]],
        [initial[1], optimized[1]],
        [initial[2], optimized[2]],
        color="#2f855a",
        linewidth=2.5,
        label="Optimization displacement",
    )
    axis.set_title("Fully Synthetic Planning Demonstration")
    axis.set_xlabel("x (synthetic units)")
    axis.set_ylabel("y (synthetic units)")
    axis.set_zlabel("z (synthetic units)")
    axis.set_box_aspect((1, 1, 1))
    axis.view_init(elev=23, azim=38)
    axis.legend(loc="upper left", fontsize=8)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_optimization_progress(scores: list[float], output_path: Path) -> None:
    """Plot running best score by evaluation index, not optimizer iteration."""
    running_best = np.maximum.accumulate(np.asarray(scores, dtype=float))
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    axis.plot(np.arange(1, len(running_best) + 1), running_best, color="#2b6cb0")
    axis.set_title("Synthetic Optimization Progress")
    axis.set_xlabel("Score evaluation index")
    axis.set_ylabel("Running best heuristic score")
    axis.grid(alpha=0.25)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    """Generate synthetic inputs, run the existing algorithm, and save figures."""
    start_time = perf_counter()
    output_dir = Path(__file__).resolve().parents[2] / "docs" / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    image, clipped, normalized = create_synthetic_ct()
    surface = create_synthetic_surface()
    seed_point = tuple(CENTER + np.array((SURFACE_RADIUS, 0.0, 0.0)))
    candidates = generate_search_sphere(seed_point, radius=2.0, step=1.0)

    diagnostics = io.StringIO()
    with redirect_stdout(diagnostics):
        recording_scorer = RecordingScorer(EntryPointScorer(image, surface))
        candidate_scores = [recording_scorer.score(point) for point in candidates]
        initial_index = int(np.argmax(candidate_scores))
        initial_point = tuple(candidates[initial_index])
        initial_score = float(candidate_scores[initial_index])

        optimizer = EntryPointOptimizer(recording_scorer, bone_mesh=surface)
        optimized_point = tuple(optimizer.optimize(initial_point))
        optimized_score = recording_scorer.score(optimized_point)

    planning_image = output_dir / "synthetic_planning_demo.png"
    progress_image = output_dir / "optimization_progress.png"
    save_planning_visualization(
        surface,
        candidates,
        initial_point,
        optimized_point,
        planning_image,
    )
    save_optimization_progress(recording_scorer.scores, progress_image)

    runtime = perf_counter() - start_time
    print("Synthetic inputs: generated in memory (no input files read)")
    print(f"HU clipped range: {float(clipped.min()):.1f} to {float(clipped.max()):.1f}")
    print(f"HU normalized range: {float(normalized.min()):.3f} to {float(normalized.max()):.3f}")
    print(f"Generated candidates: {len(candidates)}")
    print(f"Initial point: {np.round(initial_point, 4).tolist()}")
    print(f"Initial score: {initial_score:.6f}")
    print(f"Optimized point: {np.round(optimized_point, 4).tolist()}")
    print(f"Optimized score: {optimized_score:.6f}")
    print(f"Total score evaluations: {len(recording_scorer.scores)}")
    print(f"Runtime: {runtime:.3f} s")
    print(f"Planning visualization: {planning_image}")
    print(f"Optimization progress: {progress_image}")


if __name__ == "__main__":
    main()
